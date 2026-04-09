import { useEffect, useRef, useState } from 'react';
import { ChevronDown, ChevronUp, Loader2, MessageCircle, Mic, Send, X } from 'lucide-react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeSanitize from 'rehype-sanitize';

interface Message {
  role: 'user' | 'bot';
  text: string;
  isAudio?: boolean;
  citations?: Citation[];
}

interface Citation {
  title: string;
  url?: string;
  id?: string;
  score?: number;
}

type LeadStage = 'email' | 'name' | 'chat';

const DEFAULT_API_BASE_URL = import.meta.env.DEV ? 'http://localhost:8000' : '/backend';
const API_BASE_URL = (() => {
  const configuredValue = (import.meta.env.VITE_API_BASE_URL || '').trim();
  const candidate = (configuredValue || DEFAULT_API_BASE_URL).replace(/\/+$/, '');
  const pointsToLocalhost = /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?(\/.*)?$/i.test(candidate);

  // In deployed environments, localhost is unreachable from the browser and causes CORS/PNA failures.
  if (!import.meta.env.DEV && pointsToLocalhost) {
    return '/backend';
  }

  return candidate;
})();
const FLOATING_BOT_IMAGE_URL = (import.meta.env.VITE_FLOATING_BOT_IMAGE_URL || '/bot.gif').trim();

const SESSION_STORAGE_KEY = 'vtl_session_id';
const EMAIL_STORAGE_KEY = 'vtl_lead_email';
const NAME_STORAGE_KEY = 'vtl_lead_name';
const DEFAULT_SUGGESTED_QUESTIONS: string[] = [];

const normalizeQuestionText = (value: string): string => {
  return value.trim().toLowerCase().replace(/\s+/g, ' ');
};

const STARTER_QUESTION_KEYS = new Set(DEFAULT_SUGGESTED_QUESTIONS.map(normalizeQuestionText));

const decodeHeaderValue = (value: string | null): string => {
  if (!value) {
    return '';
  }

  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
};

const normalizeCitationUrl = (value: unknown): string | undefined => {
  if (typeof value !== 'string') {
    return undefined;
  }

  let url = value.trim();
  if (!url) {
    return undefined;
  }

  if (/^www\./i.test(url)) {
    url = `https://${url}`;
  } else if (!/^https?:\/\//i.test(url)) {
    if (/^[a-z0-9.-]+\.[a-z]{2,}\//i.test(url)) {
      url = `https://${url}`;
    } else {
      return undefined;
    }
  }

  return url.replace(/ /g, '%20');
};

const buildTraceErrorMessage = (message: string, traceId: string = ''): string => {
  const cleaned = message.trim() || 'Sorry, an error occurred while streaming the response.';
  if (!traceId) {
    return cleaned;
  }
  return `${cleaned} (Trace ID: ${traceId})`;
};

const normalizeMarkdownText = (text: string): string => {
  const normalized = (text || '').replace(/\r\n?/g, '\n');
  const fenceCount = (normalized.match(/(^|\n)```/g) || []).length;
  if (fenceCount % 2 === 1) {
    return `${normalized}\n\n\`\`\``;
  }
  return normalized;
};

function MarkdownMessage({ text }: { text: string }) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]} className="vtl-markdown">
      {normalizeMarkdownText(text)}
    </ReactMarkdown>
  );
}

const isValidEmail = (email: string): boolean => {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());
};

const createSessionId = (): string => {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID().replace(/[^a-zA-Z0-9_-]/g, '').slice(0, 64);
  }
  return `session_${Date.now()}`;
};

function App() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isStreamingResponse, setIsStreamingResponse] = useState(false);
  const [isWaitingForFirstToken, setIsWaitingForFirstToken] = useState(false);
  const [floatingImageError, setFloatingImageError] = useState(false);

  const [sessionId, setSessionId] = useState('');
  const [leadEmail, setLeadEmail] = useState('');
  const [leadName, setLeadName] = useState('');
  const [leadStage, setLeadStage] = useState<LeadStage>('email');
  const [dynamicSuggestedQuestions, setDynamicSuggestedQuestions] = useState<string[]>([]);
  const [hasStartedChat, setHasStartedChat] = useState(false);
  const [showSuggestedQuestions, setShowSuggestedQuestions] = useState(true);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const storedSessionId = localStorage.getItem(SESSION_STORAGE_KEY)?.trim();
    const storedEmail = localStorage.getItem(EMAIL_STORAGE_KEY)?.trim() || '';
    const storedName = localStorage.getItem(NAME_STORAGE_KEY)?.trim() || '';

    const resolvedSessionId = storedSessionId || createSessionId();
    setSessionId(resolvedSessionId);
    localStorage.setItem(SESSION_STORAGE_KEY, resolvedSessionId);

    setLeadEmail(storedEmail);
    setLeadName(storedName);

    if (storedEmail && storedName) {
      setLeadStage('chat');
      setHasStartedChat(true);
      setMessages([
        {
          role: 'bot',
          text: `Welcome back ${storedName}! How can I help you today?`,
        },
      ]);
      return;
    }

    if (storedEmail) {
      setLeadStage('name');
      setMessages([
        { role: 'bot', text: 'Please share your full name to continue.' },
      ]);
      return;
    }

    setLeadStage('email');
    setMessages([
      {
        role: 'bot',
        text: 'Hi! Before we begin, please share your email address.',
      },
    ]);
  }, []);

  useEffect(() => {
    if (leadEmail) {
      localStorage.setItem(EMAIL_STORAGE_KEY, leadEmail);
    }
  }, [leadEmail]);

  useEffect(() => {
    if (leadName) {
      localStorage.setItem(NAME_STORAGE_KEY, leadName);
    }
  }, [leadName]);

  useEffect(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading]);

  const selectDynamicSuggestions = (rawSuggestions: unknown, currentPrompt: string = ''): string[] => {
    const currentPromptKey = normalizeQuestionText(currentPrompt);
    const seenKeys = new Set<string>();

    if (!Array.isArray(rawSuggestions)) {
      return [];
    }

    return rawSuggestions
      .filter((item: unknown): item is string => typeof item === 'string' && item.trim().length > 0)
      .filter((item: string) => {
        const normalized = normalizeQuestionText(item);
        if (!normalized || STARTER_QUESTION_KEYS.has(normalized)) {
          return false;
        }
        if (currentPromptKey && normalized === currentPromptKey) {
          return false;
        }
        if (seenKeys.has(normalized)) {
          return false;
        }
        seenKeys.add(normalized);
        return true;
      })
      .slice(0, 2);
  };

  const appendToLatestBotMessage = (chunk: string) => {
    if (!chunk) {
      return;
    }

    setMessages((prev) => {
      const next = [...prev];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === 'bot') {
          next[index] = { ...next[index], text: `${next[index].text}${chunk}` };
          return next;
        }
      }
      return [...next, { role: 'bot', text: chunk }];
    });
  };

  const setLatestBotMessageText = (text: string) => {
    setMessages((prev) => {
      const next = [...prev];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === 'bot') {
          next[index] = { ...next[index], text };
          return next;
        }
      }
      return [...next, { role: 'bot', text }];
    });
  };

  const setLatestBotMessageCitations = (citations: Citation[]) => {
    setMessages((prev) => {
      const next = [...prev];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === 'bot') {
          next[index] = { ...next[index], citations };
          return next;
        }
      }
      return next;
    });
  };

  const refreshSuggestedQuestions = async (targetSessionId: string = sessionId, currentPrompt: string = '') => {
    if (!targetSessionId) {
      return;
    }

    try {
      const response = await axios.get(`${API_BASE_URL}/api/chat/suggestions`, {
        params: {
          session_id: targetSessionId,
          limit: 2,
        },
      });
      setDynamicSuggestedQuestions(selectDynamicSuggestions(response.data?.suggestions, currentPrompt));
    } catch {
      setDynamicSuggestedQuestions([]);
    }
  };

  const visibleSuggestedQuestions = hasStartedChat
    ? dynamicSuggestedQuestions
    : DEFAULT_SUGGESTED_QUESTIONS;

  useEffect(() => {
    if (leadStage === 'chat' && sessionId && hasStartedChat) {
      void refreshSuggestedQuestions(sessionId);
    }
  }, [leadStage, sessionId, hasStartedChat]);

  const submitUserMessage = async (rawMessage: string) => {
    if (!rawMessage.trim()) {
      return;
    }

    const userMsg = rawMessage.trim();
    setMessages((prev) => [...prev, { role: 'user', text: userMsg }]);

    if (leadStage === 'email') {
      if (!isValidEmail(userMsg)) {
        setMessages((prev) => [
          ...prev,
          {
            role: 'bot',
            text: 'That email looks invalid. Please enter a valid email address.',
          },
        ]);
        return;
      }

      const normalizedEmail = userMsg.toLowerCase();
      setLeadEmail(normalizedEmail);
      setLeadStage('name');
      setMessages((prev) => [
        ...prev,
        {
          role: 'bot',
          text: 'Thanks! Now please share your name.',
        },
      ]);
      return;
    }

    if (leadStage === 'name') {
      const normalizedName = userMsg.replace(/\s+/g, ' ').trim();
      if (normalizedName.length < 2) {
        setMessages((prev) => [
          ...prev,
          {
            role: 'bot',
            text: 'Please enter your full name to continue.',
          },
        ]);
        return;
      }

      setLeadName(normalizedName);
      setLeadStage('chat');
      setMessages((prev) => [
        ...prev,
        {
          role: 'bot',
          text: `Nice to meet you, ${normalizedName}. How can I help you today?`,
        },
      ]);
      return;
    }

    setHasStartedChat(true);
    setIsLoading(true);
    setIsStreamingResponse(true);
    setIsWaitingForFirstToken(true);
    setMessages((prev) => [...prev, { role: 'bot', text: '' }]);

    try {
      const formData = new FormData();
      formData.append('query', userMsg);
      formData.append('session_id', sessionId);
      formData.append('lead_email', leadEmail);
      formData.append('lead_name', leadName);

      const response = await fetch(`${API_BASE_URL}/api/chat/text/stream`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok || !response.body) {
        const responseTraceId = response.headers.get('X-Trace-Id') || '';
        let detailMessage = 'Streaming API failed';
        try {
          const payload = await response.json();
          if (payload && typeof payload === 'object') {
            const detail = (payload as { detail?: unknown }).detail;
            if (typeof detail === 'string' && detail.trim()) {
              detailMessage = detail;
            }
          }
        } catch {
          // Ignore JSON parse errors and keep fallback message.
        }
        throw new Error(buildTraceErrorMessage(detailMessage, responseTraceId));
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let streamedText = '';
      let resolvedSessionId = sessionId;
      let resolvedLeadEmail = leadEmail;
      let resolvedLeadName = leadName;
      let streamSuggestions: string[] = [];
      let streamCitations: Citation[] = [];

      const processEvent = (eventType: string, payload: string) => {
        let parsed: unknown;
        try {
          parsed = JSON.parse(payload);
        } catch {
          return;
        }

        if (!parsed || typeof parsed !== 'object') {
          return;
        }

        const data = parsed as {
          token?: unknown;
          reply?: unknown;
          session_id?: unknown;
          lead?: unknown;
          suggestions?: unknown;
          message?: unknown;
          trace_id?: unknown;
          citations?: unknown;
        };

        if (eventType === 'token') {
          const token = typeof data.token === 'string' ? data.token : '';
          if (token) {
            setIsWaitingForFirstToken(false);
            streamedText += token;
            appendToLatestBotMessage(token);
          }
          return;
        }

        if (eventType === 'done') {
          setIsWaitingForFirstToken(false);
          const doneReply = typeof data.reply === 'string' ? data.reply : '';
          if (!streamedText.trim() && doneReply) {
            streamedText = doneReply;
            setLatestBotMessageText(doneReply);
          }

          if (typeof data.session_id === 'string' && data.session_id.trim()) {
            resolvedSessionId = data.session_id.trim();
          }

          if (data.lead && typeof data.lead === 'object') {
            const lead = data.lead as { email?: unknown; name?: unknown };
            if (typeof lead.email === 'string') {
              resolvedLeadEmail = lead.email;
            }
            if (typeof lead.name === 'string') {
              resolvedLeadName = lead.name;
            }
          }

          streamSuggestions = selectDynamicSuggestions(data.suggestions, userMsg);

          if (Array.isArray(data.citations)) {
            streamCitations = data.citations
              .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object')
              .map((item) => ({
                title: typeof item.title === 'string' && item.title.trim()
                  ? item.title.trim()
                  : 'Source document',
                url: normalizeCitationUrl(item.url),
                id: typeof item.id === 'string' && item.id.trim() ? item.id.trim() : undefined,
                score: typeof item.score === 'number' ? item.score : undefined,
              }))
              .slice(0, 5);
          }
          return;
        }

        if (eventType === 'error') {
          const traceId = typeof data.trace_id === 'string' ? data.trace_id : '';
          const errorMessage = typeof data.message === 'string'
            ? data.message
            : 'Sorry, an error occurred while streaming the response.';
          throw new Error(buildTraceErrorMessage(errorMessage, traceId));
        }
      };

      const processRawEvent = (rawEvent: string) => {
        const lines = rawEvent.replace(/\r/g, '').split('\n');
        let eventType = 'message';
        const dataLines: string[] = [];

        for (const line of lines) {
          if (line.startsWith('event:')) {
            eventType = line.slice(6).trim();
            continue;
          }
          if (line.startsWith('data:')) {
            dataLines.push(line.slice(5).trim());
          }
        }

        if (dataLines.length > 0) {
          processEvent(eventType, dataLines.join('\n'));
        }
      };

      const drainEventBuffer = (flushRemainder: boolean = false) => {
        let splitIndex = buffer.indexOf('\n\n');
        while (splitIndex !== -1) {
          const rawEvent = buffer.slice(0, splitIndex);
          buffer = buffer.slice(splitIndex + 2);
          processRawEvent(rawEvent);
          splitIndex = buffer.indexOf('\n\n');
        }

        if (flushRemainder && buffer.trim()) {
          processRawEvent(buffer);
          buffer = '';
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          buffer += decoder.decode();
          drainEventBuffer(true);
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        drainEventBuffer();
      }

      if (!streamedText.trim()) {
        throw new Error('Empty streamed response');
      }

      if (streamCitations.length > 0) {
        setLatestBotMessageCitations(streamCitations);
      }

      if (resolvedSessionId !== sessionId) {
        setSessionId(resolvedSessionId);
        localStorage.setItem(SESSION_STORAGE_KEY, resolvedSessionId);
      }
      if (resolvedLeadEmail && resolvedLeadEmail !== leadEmail) {
        setLeadEmail(resolvedLeadEmail);
      }
      if (resolvedLeadName && resolvedLeadName !== leadName) {
        setLeadName(resolvedLeadName);
      }

      if (streamSuggestions.length > 0) {
        setDynamicSuggestedQuestions(streamSuggestions);
      } else {
        void refreshSuggestedQuestions(resolvedSessionId, userMsg);
      }
    } catch (error) {
      const errorMessage = error instanceof Error && error.message
        ? error.message
        : 'Sorry, an error occurred while streaming the response.';
      setLatestBotMessageText(errorMessage);
    } finally {
      setIsWaitingForFirstToken(false);
      setIsStreamingResponse(false);
      setIsLoading(false);
    }
  };

  const handleTextSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputText.trim()) {
      return;
    }

    const messageToSend = inputText;
    setInputText('');
    await submitUserMessage(messageToSend);
  };

  const handleSuggestedQuestionClick = async (question: string) => {
    if (leadStage !== 'chat' || isLoading || isRecording) {
      return;
    }

    await submitUserMessage(question);
  };

  const startRecording = async () => {
    if (isLoading || isRecording || leadStage !== 'chat') {
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event: BlobEvent) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = handleAudioStop;
      mediaRecorder.start();
      setIsRecording(true);
    } catch {
      alert('Please allow microphone access to use voice features.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
    }
  };

  const handleAudioStop = async () => {
    setIsLoading(true);
    const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
    const audioFile = new File([audioBlob], 'recording.webm', { type: 'audio/webm' });

    const formData = new FormData();
    formData.append('audio', audioFile);

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat/voice`, {
        method: 'POST',
        headers: {
          'X-Session-Id': sessionId,
          'X-Lead-Email': leadEmail,
          'X-Lead-Name': leadName,
        },
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Voice API failed');
      }

      let userQuery = decodeHeaderValue(response.headers.get('X-User-Query-Encoded'))
        || response.headers.get('X-User-Query')
        || 'Voice Message';
      let botReply = decodeHeaderValue(response.headers.get('X-Bot-Reply-Encoded'))
        || response.headers.get('X-Bot-Reply')
        || 'Audio Reply';

      try {
        const lastTurnResponse = await axios.get(`${API_BASE_URL}/api/chat/last`, {
          params: { session_id: sessionId },
        });
        userQuery = lastTurnResponse.data.user_query || userQuery;
        botReply = lastTurnResponse.data.reply || botReply;
      } catch {
        // Keep header-based fallbacks when last-turn lookup is unavailable.
      }

      setMessages((prev) => [
        ...prev,
        { role: 'user', text: userQuery, isAudio: true },
        { role: 'bot', text: botReply, isAudio: true },
      ]);
      setHasStartedChat(true);
      void refreshSuggestedQuestions(sessionId, userQuery);

      const audioResponseBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioResponseBlob);
      const audio = new Audio(audioUrl);
      await audio.play();
      audio.onended = () => URL.revokeObjectURL(audioUrl);
    } catch {
      setMessages((prev) => [...prev, { role: 'bot', text: 'Sorry, failed to process audio.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  const voiceHintText = isRecording
    ? '🔴 Recording... release to send'
    : leadStage === 'chat'
      ? 'Hold mic to record • Release to send'
      : 'Complete email and name first to enable voice';

  const showFloatingImage = !isOpen && !!FLOATING_BOT_IMAGE_URL && !floatingImageError;

  return (
    <>
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`fixed z-50 transition-transform hover:scale-105 flex items-center justify-center ${
          isOpen ? 'top-3 right-3 sm:top-auto sm:bottom-6 sm:right-6' : 'bottom-4 right-4 sm:bottom-6 sm:right-6'
        } ${
          showFloatingImage
            ? 'w-16 h-16 sm:w-[110px] sm:h-[110px] rounded-full bg-transparent shadow-none overflow-hidden p-0'
            : 'p-3 sm:p-4 vtl-brand-gradient text-white rounded-full shadow-2xl hover:brightness-95'
        }`}
      >
        {isOpen ? (
          <X className="w-5 h-5 sm:w-6 sm:h-6" />
        ) : showFloatingImage ? (
          <img
            src={FLOATING_BOT_IMAGE_URL}
            alt="Assistant"
            className="w-full h-full object-cover object-center rounded-full"
            onError={() => setFloatingImageError(true)}
          />
        ) : (
          <MessageCircle className="w-5 h-5 sm:w-6 sm:h-6" />
        )}
      </button>

      {isOpen && (
        <div className="fixed inset-x-0 top-0 bottom-0 sm:inset-auto sm:bottom-24 sm:right-6 z-40 w-full sm:w-[min(540px,94vw)] h-full sm:h-[760px] sm:max-h-[85vh] bg-[var(--vtl-panel)] rounded-none sm:rounded-2xl shadow-2xl flex flex-col border border-[var(--vtl-border)] overflow-hidden">
          <div className="vtl-brand-gradient p-3 sm:p-4 text-white font-bold text-base sm:text-lg flex justify-between items-center shadow-md z-10">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 bg-[var(--vtl-accent)] rounded-full animate-pulse"></div>
              <span>Desire Assistant</span>
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-3 sm:p-4 space-y-4 bg-[var(--vtl-surface)]">
            {messages.map((msg, idx) => (
              <div key={`${msg.role}-${idx}`} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`p-3 rounded-2xl text-sm max-w-[90%] sm:max-w-[85%] shadow-sm bg-[var(--vtl-panel)] border border-[var(--vtl-border)] text-[var(--vtl-text)] ${
                    msg.role === 'user' ? 'rounded-br-none' : 'rounded-bl-none'
                  }`}
                >
                  {msg.isAudio && <span className="text-xs opacity-75 block mb-1">🎤 Voice</span>}
                  {msg.role === 'bot' && msg.text.trim().length === 0 && isStreamingResponse && isLoading && isWaitingForFirstToken && idx === messages.length - 1 ? (
                    <div className="flex items-center gap-2 text-[var(--vtl-muted)]">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span>Thinking...</span>
                    </div>
                  ) : msg.role === 'bot' ? (
                    <MarkdownMessage text={msg.text} />
                  ) : (
                    <span className="whitespace-pre-wrap break-words">{msg.text}</span>
                  )}

                  {msg.role === 'bot' && Array.isArray(msg.citations) && msg.citations.length > 0 && (
                    <div className="mt-3 border-t border-[var(--vtl-border)] pt-2">
                      <div className="text-xs font-semibold text-[var(--vtl-muted)] mb-1">Sources</div>
                      <ul className="space-y-1 text-xs">
                        {msg.citations.map((citation: Citation, citationIndex: number) => {
                          const label = citation.title || `Source ${citationIndex + 1}`;
                          if (citation.url) {
                            return (
                              <li key={`${label}-${citationIndex}`}>
                                <a
                                  href={citation.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-[var(--vtl-primary)] underline break-all"
                                >
                                  {label}
                                </a>
                              </li>
                            );
                          }

                          return (
                            <li key={`${label}-${citationIndex}`} className="text-[var(--vtl-muted)] break-all">
                              {label}
                            </li>
                          );
                        })}
                      </ul>
                    </div>
                  )}
                </div>
              </div>
            ))}

            {isLoading && !isStreamingResponse && (
              <div className="flex justify-start">
                <div className="bg-[var(--vtl-panel)] border border-[var(--vtl-border)] p-3 rounded-2xl rounded-bl-none flex items-center gap-2 text-[var(--vtl-muted)] text-sm">
                  <Loader2 className="w-4 h-4 animate-spin" /> Thinking...
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="p-3 bg-[var(--vtl-panel)] border-t border-[var(--vtl-border)]">
            {leadStage === 'chat' && (
              <div className="mb-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="text-xs text-[var(--vtl-muted)]">Suggested questions</div>
                  <button
                    type="button"
                    onClick={() => setShowSuggestedQuestions((prev) => !prev)}
                    className="p-1 rounded-full text-[var(--vtl-primary)] hover:bg-[var(--vtl-chip-bg)]"
                    aria-label={showSuggestedQuestions ? 'Hide suggested questions' : 'Show suggested questions'}
                    title={showSuggestedQuestions ? 'Hide suggestions' : 'Show suggestions'}
                  >
                    {showSuggestedQuestions ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
                  </button>
                </div>

                {showSuggestedQuestions && (
                  <div className="flex flex-wrap gap-2">
                    {visibleSuggestedQuestions.map((question) => (
                      <button
                        key={question}
                        type="button"
                        onClick={() => void handleSuggestedQuestionClick(question)}
                        disabled={isLoading || isRecording}
                        className="px-3 py-1.5 rounded-full text-xs bg-[var(--vtl-chip-bg)] text-[var(--vtl-primary)] hover:bg-[var(--vtl-chip-hover)] disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {question}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}

            <div className={`text-xs mb-2 ${isRecording ? 'text-red-500 font-medium' : 'text-[var(--vtl-muted)]'}`}>
              {voiceHintText}
            </div>

            <div className="flex items-center gap-2">
            <button
              onMouseDown={startRecording}
              onMouseUp={stopRecording}
              onMouseLeave={stopRecording}
              onTouchStart={startRecording}
              onTouchEnd={stopRecording}
              title="Hold to record voice message, release to send"
              className={`p-2 sm:p-2.5 rounded-full flex-shrink-0 ${
                isRecording
                  ? 'bg-red-500 text-white animate-pulse'
                  : 'bg-[var(--vtl-chip-bg)] text-[var(--vtl-primary)] hover:bg-[var(--vtl-chip-hover)] disabled:opacity-50 disabled:cursor-not-allowed'
              }`}
              disabled={leadStage !== 'chat' || isLoading}
            >
              <Mic className="w-4 h-4 sm:w-5 sm:h-5" />
            </button>

            <form onSubmit={handleTextSubmit} className="flex-1 flex gap-2">
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder={
                  leadStage === 'email'
                    ? 'Enter your email address'
                    : leadStage === 'name'
                      ? 'Enter your full name'
                      : 'Message...'
                }
                className="flex-1 min-w-0 px-3 sm:px-4 py-2 text-sm rounded-full bg-[var(--vtl-chip-bg)] text-[var(--vtl-text)] border border-transparent focus:bg-white focus:border-[var(--vtl-secondary)] outline-none"
                disabled={isRecording || isLoading}
              />
              <button
                type="submit"
                title="Send message"
                disabled={!inputText.trim() || isRecording || isLoading}
                className="p-2 sm:p-2.5 vtl-brand-gradient text-white rounded-full hover:brightness-95 disabled:opacity-50"
              >
                <Send className="w-4 h-4" />
              </button>
            </form>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default App;
