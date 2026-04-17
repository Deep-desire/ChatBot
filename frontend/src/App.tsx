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
  videos?: VideoSource[];
}

interface Citation {
  title: string;
  url?: string;
  id?: string;
  score?: number;
}

interface VideoSource {
  title: string;
  url: string;
  embedUrl: string;
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
const DYNAMIC_SUGGESTION_MAX_CHARS = 84;
const DEFAULT_SUGGESTED_QUESTIONS: string[] = [
  'What is Desire Infoweb?',
  'What type of services does Desire Infoweb provide?',
  'What type of AI projects has Desire Infoweb completed?',
];

const normalizeQuestionText = (value: string): string => {
  return value.trim().toLowerCase().replace(/\s+/g, ' ');
};

const clampSuggestionLength = (value: string): string => {
  const normalized = value.trim().replace(/\s+/g, ' ');
  if (normalized.length <= DYNAMIC_SUGGESTION_MAX_CHARS) {
    return normalized;
  }
  const sliced = normalized.slice(0, DYNAMIC_SUGGESTION_MAX_CHARS - 1).trim();
  const boundary = sliced.lastIndexOf(' ');
  const compact = boundary >= 28 ? sliced.slice(0, boundary).trim() : sliced;
  return `${compact}?`;
};

const STARTER_QUESTION_KEYS = new Set(DEFAULT_SUGGESTED_QUESTIONS.map(normalizeQuestionText));
const LOW_SIGNAL_SUGGESTION_KEYS = new Set([
  'hi',
  'hii',
  'hiii',
  'hello',
  'hey',
  'ok',
  'okay',
  'thanks',
  'thankyou',
  'thank you',
]);

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

const repairMalformedDocumentUrl = (value: string): string => {
  if (!value) {
    return value;
  }

  const malformedDocPattern = /(\.(?:pdf|doc|docx|txt|md|csv|xls|xlsx|ppt|pptx|html|htm|json))\d+$/i;

  try {
    const parsed = new URL(value);
    parsed.pathname = parsed.pathname.replace(malformedDocPattern, '$1');
    return parsed.toString();
  } catch {
    return value.replace(malformedDocPattern, '$1');
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

  return repairMalformedDocumentUrl(url.replace(/ /g, '%20'));
};

const getCitationLabel = (citation: Citation, citationIndex: number): string => {
  const title = (citation.title || '').trim();
  const normalizedUrl = normalizeCitationUrl(citation.url);

  if (normalizedUrl) {
    try {
      const parsed = new URL(normalizedUrl);
      const fileName = decodeURIComponent(parsed.pathname.split('/').pop() || '').trim();
      if (fileName && /\.pdf$/i.test(fileName)) {
        return fileName;
      }
    } catch {
      // Ignore parse errors and fall through to URL display.
    }

    return normalizedUrl;
  }

  if (title && !/^https?:\/\//i.test(title)) {
    return title;
  }

  return title || `Source ${citationIndex + 1}`;
};

const buildVideoEmbedUrl = (value: unknown): string | undefined => {
  const normalized = normalizeCitationUrl(value);
  if (!normalized) {
    return undefined;
  }

  try {
    const parsed = new URL(normalized);
    const host = parsed.hostname.toLowerCase();
    const path = parsed.pathname;

    if (host === 'youtu.be') {
      const videoId = path.replace(/^\/+/, '').split('/')[0];
      return videoId ? `https://www.youtube.com/embed/${videoId}` : undefined;
    }

    if (host.includes('youtube.com')) {
      if (path === '/watch') {
        const videoId = parsed.searchParams.get('v') || '';
        return videoId ? `https://www.youtube.com/embed/${videoId}` : undefined;
      }
      if (path.startsWith('/shorts/')) {
        const videoId = path.split('/')[2] || '';
        return videoId ? `https://www.youtube.com/embed/${videoId}` : undefined;
      }
      if (path.startsWith('/embed/')) {
        return normalized;
      }
    }

    if (host.includes('vimeo.com')) {
      if (host.startsWith('player.') && path.startsWith('/video/')) {
        return normalized;
      }
      const match = path.match(/\/(\d+)/);
      return match ? `https://player.vimeo.com/video/${match[1]}` : undefined;
    }

    if (/\.(mp4|webm|mov|m3u8)(\?|$)/i.test(normalized)) {
      return normalized;
    }
  } catch {
    return undefined;
  }

  return undefined;
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

const looksAbruptlyTruncated = (text: string): boolean => {
  const value = (text || '').trim();
  
  // Empty text is truncated
  if (!value) {
    return true;
  }
  
  // Ends cleanly with punctuation
  if (/[.!?]$/.test(value)) {
    return false;
  }
  
  // Ends with connector words (incomplete sentence)
  if (/\b(and|or|to|for|with|but|nor|yet|by|as|if|is|was|are|been|being)\s*$/.test(value)) {
    return true;
  }
  
  // Check last line pattern
  const lastLine = value.split('\n').pop()?.trim() || '';
  if (!lastLine) {
    return false;
  }
  
  // Last line is just a heading start (incomplete markdown)
  if (/^#{1,6}\s+$/.test(lastLine)) {
    return true;
  }
  
  // Last line is just a list marker (incomplete list item)
  if (/^[-*]\s*$/.test(lastLine)) {
    return true;
  }
  
  // Last line is just a colon or dash (incomplete continuation)
  if (/[:–—-]\s*$/.test(lastLine) && lastLine.length < 5) {
    return true;
  }
  
  // Otherwise, consider it complete (be optimistic about streaming)
  return false;
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
  const speechRecognitionRef = useRef<any>(null);
  const voiceDraftTranscriptRef = useRef('');
  const audioChunksRef = useRef<Blob[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const streamCharacterQueueRef = useRef('');
  const streamTypeTimerRef = useRef<number | null>(null);
  const streamDrainResolverRef = useRef<(() => void) | null>(null);

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
      // Show starter questions on each fresh widget load; switch to dynamic after first new prompt.
      setHasStartedChat(false);
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
      messagesEndRef.current.scrollIntoView({ behavior: isStreamingResponse ? 'auto' : 'smooth' });
    }
  }, [messages, isLoading, isStreamingResponse]);

  const invokeAndClearStreamDrainResolver = () => {
    const resolver = streamDrainResolverRef.current as (() => void) | null;
    streamDrainResolverRef.current = null;
    if (resolver) {
      resolver();
    }
  };

  useEffect(() => {
    return () => {
      if (streamTypeTimerRef.current !== null) {
        window.clearTimeout(streamTypeTimerRef.current);
        streamTypeTimerRef.current = null;
      }
      streamCharacterQueueRef.current = '';
      invokeAndClearStreamDrainResolver();
    };
  }, []);

  const selectDynamicSuggestions = (rawSuggestions: unknown, currentPrompt: string = ''): string[] => {
    const currentPromptKey = normalizeQuestionText(currentPrompt);
    const seenKeys = new Set<string>();
    const askedQuestionKeys = new Set(
      messages
        .filter((message) => message.role === 'user')
        .map((message) => normalizeQuestionText(message.text)),
    );

    if (!Array.isArray(rawSuggestions)) {
      return [];
    }

    return rawSuggestions
      .filter((item: unknown): item is string => typeof item === 'string' && item.trim().length > 0)
      .map((item: string) => clampSuggestionLength(item))
      .filter((item: string) => {
        const normalized = normalizeQuestionText(item);
        if (!normalized || STARTER_QUESTION_KEYS.has(normalized)) {
          return false;
        }
        if (askedQuestionKeys.has(normalized)) {
          return false;
        }
        if (LOW_SIGNAL_SUGGESTION_KEYS.has(normalized) || normalized.length < 8) {
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
      .slice(0, 3);
  };

  const appendToLatestBotMessage = (chunk: string) => {
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

  const resolveStreamDrainIfIdle = () => {
    if (streamTypeTimerRef.current !== null || streamCharacterQueueRef.current.length > 0) {
      return;
    }
    invokeAndClearStreamDrainResolver();
  };

  const getCharacterDelay = (character: string, queueLength: number): number => {
    const baseDelay = queueLength > 240 ? 6 : queueLength > 120 ? 8 : 12;
    if (/[.!?]/.test(character)) {
      return baseDelay + 38;
    }
    if (/[,;:]/.test(character)) {
      return baseDelay + 20;
    }
    if (/\s/.test(character)) {
      return baseDelay + 6;
    }
    return baseDelay;
  };

  const flushRemainingStreamCharacters = () => {
    if (!streamCharacterQueueRef.current) {
      return;
    }
    const remaining = streamCharacterQueueRef.current;
    streamCharacterQueueRef.current = '';
    appendToLatestBotMessage(remaining);
  };

  const typeNextStreamCharacter = () => {
    if (!streamCharacterQueueRef.current) {
      streamTypeTimerRef.current = null;
      resolveStreamDrainIfIdle();
      return;
    }

    const nextCharacter = streamCharacterQueueRef.current[0];
    streamCharacterQueueRef.current = streamCharacterQueueRef.current.slice(1);
    appendToLatestBotMessage(nextCharacter);

    const delay = getCharacterDelay(nextCharacter, streamCharacterQueueRef.current.length);
    streamTypeTimerRef.current = window.setTimeout(typeNextStreamCharacter, delay);
  };

  const queueStreamToken = (token: string) => {
    if (!token) {
      return;
    }
    streamCharacterQueueRef.current += token;
    if (streamTypeTimerRef.current === null) {
      typeNextStreamCharacter();
    }
  };

  const waitForStreamAnimationDrain = async (timeoutMs?: number) => {
    if (streamTypeTimerRef.current === null && streamCharacterQueueRef.current.length === 0) {
      return;
    }

    const queueLength = streamCharacterQueueRef.current.length;
    const computedTimeout = timeoutMs ?? Math.min(18000, Math.max(3000, queueLength * 10));

    await new Promise<void>((resolve) => {
      const fallbackTimer = window.setTimeout(() => {
        if (streamDrainResolverRef.current === onDrain) {
          streamDrainResolverRef.current = null;
        }
        // Ensure remaining queued characters are not lost on timeout.
        if (streamTypeTimerRef.current !== null) {
          window.clearTimeout(streamTypeTimerRef.current);
          streamTypeTimerRef.current = null;
        }
        flushRemainingStreamCharacters();
        resolve();
      }, computedTimeout);

      const onDrain = () => {
        window.clearTimeout(fallbackTimer);
        resolve();
      };

      streamDrainResolverRef.current = onDrain;
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

  const setLatestUserMessageText = (text: string) => {
    setMessages((prev) => {
      const next = [...prev];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === 'user') {
          next[index] = { ...next[index], text, isAudio: true };
          return next;
        }
      }
      return [...next, { role: 'user', text, isAudio: true }];
    });
  };

  const insertUserMessageBeforeLatestVoiceBot = (text: string) => {
    setMessages((prev) => {
      const next = [...prev];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        const message = next[index];
        if (message.role === 'bot' && message.isAudio && message.text.trim().length === 0) {
          next.splice(index, 0, { role: 'user', text, isAudio: true });
          return next;
        }
      }
      return [...next, { role: 'user', text, isAudio: true }];
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

  const setLatestBotMessageVideos = (videos: VideoSource[]) => {
    setMessages((prev) => {
      const next = [...prev];
      for (let index = next.length - 1; index >= 0; index -= 1) {
        if (next[index].role === 'bot') {
          next[index] = { ...next[index], videos };
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
          limit: 3,
        },
      });
      setDynamicSuggestedQuestions(selectDynamicSuggestions(response.data?.suggestions, currentPrompt));
    } catch {
      setDynamicSuggestedQuestions([]);
    }
  };

  const visibleSuggestedQuestions = hasStartedChat
    ? dynamicSuggestedQuestions.slice(0, 3)
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
    streamCharacterQueueRef.current = '';
    if (streamTypeTimerRef.current !== null) {
      window.clearTimeout(streamTypeTimerRef.current);
      streamTypeTimerRef.current = null;
    }
    invokeAndClearStreamDrainResolver();
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
      let streamVideos: VideoSource[] = [];
      let doneReplyFromServer = '';

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
          videos?: unknown;
        };

        if (eventType === 'token') {
          const token = typeof data.token === 'string' ? data.token : '';
          if (token) {
            setIsWaitingForFirstToken(false);
            streamedText += token;
            queueStreamToken(token);
          }
          return;
        }

        if (eventType === 'done') {
          setIsWaitingForFirstToken(false);
          const doneReply = typeof data.reply === 'string' ? data.reply : '';
          doneReplyFromServer = doneReply;
          if (!streamedText.trim() && doneReply) {
            streamedText = doneReply;
            queueStreamToken(doneReply);
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

          if (Array.isArray(data.videos)) {
            streamVideos = data.videos
              .filter((item): item is Record<string, unknown> => !!item && typeof item === 'object')
              .map((item) => {
                const url = normalizeCitationUrl(item.url);
                const embedFromPayload = normalizeCitationUrl(item.embed_url);
                const embedUrl = embedFromPayload || buildVideoEmbedUrl(url);
                if (!url || !embedUrl) {
                  return null;
                }

                return {
                  title: typeof item.title === 'string' && item.title.trim() ? item.title.trim() : 'Video source',
                  url,
                  embedUrl,
                };
              })
              .filter((item): item is VideoSource => !!item)
              .slice(0, 1);
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
          buffer += decoder.decode().replace(/\r/g, '');
          drainEventBuffer(true);
          break;
        }

        buffer += decoder.decode(value, { stream: true }).replace(/\r/g, '');
        drainEventBuffer();
      }

      if (!streamedText.trim()) {
        throw new Error('Empty streamed response');
      }

      await waitForStreamAnimationDrain();

      if (doneReplyFromServer.trim()) {
        const streamedTrimmed = streamedText.trim();
        const doneTrimmed = doneReplyFromServer.trim();
        if (streamedTrimmed !== doneTrimmed || looksAbruptlyTruncated(streamedTrimmed)) {
          setLatestBotMessageText(doneReplyFromServer);
          streamedText = doneReplyFromServer;
        }
      }

      if (streamCitations.length > 0) {
        setLatestBotMessageCitations(streamCitations);
      }

      if (streamVideos.length > 0) {
        setLatestBotMessageVideos(streamVideos);
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
      if (streamTypeTimerRef.current !== null) {
        window.clearTimeout(streamTypeTimerRef.current);
        streamTypeTimerRef.current = null;
      }
      flushRemainingStreamCharacters();
      const errorMessage = error instanceof Error && error.message
        ? error.message
        : 'Sorry, an error occurred while streaming the response.';
      setLatestBotMessageText(errorMessage);
    } finally {
      if (streamTypeTimerRef.current !== null) {
        window.clearTimeout(streamTypeTimerRef.current);
        streamTypeTimerRef.current = null;
      }
      flushRemainingStreamCharacters();
      streamCharacterQueueRef.current = '';
      invokeAndClearStreamDrainResolver();
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
      voiceDraftTranscriptRef.current = '';
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      const speechRecognitionFactory = (
        window as Window & {
          SpeechRecognition?: new () => any;
          webkitSpeechRecognition?: new () => any;
        }
      );
      const SpeechRecognitionCtor = speechRecognitionFactory.SpeechRecognition || speechRecognitionFactory.webkitSpeechRecognition;
      if (SpeechRecognitionCtor) {
        const recognition = new SpeechRecognitionCtor();
        recognition.lang = 'en-US';
        recognition.interimResults = true;
        recognition.continuous = true;
        recognition.onresult = (event: any) => {
          const capturedParts: string[] = [];
          const results = event?.results;
          if (!results) {
            return;
          }
          for (let index = 0; index < results.length; index += 1) {
            const transcriptPart = results[index]?.[0]?.transcript;
            if (typeof transcriptPart === 'string' && transcriptPart.trim()) {
              capturedParts.push(transcriptPart.trim());
            }
          }
          if (capturedParts.length > 0) {
            voiceDraftTranscriptRef.current = capturedParts.join(' ').replace(/\s+/g, ' ').trim();
          }
        };
        recognition.onerror = () => {
          // Ignore browser speech recognition errors; backend transcription remains the source of truth.
        };
        speechRecognitionRef.current = recognition;
        recognition.start();
      } else {
        speechRecognitionRef.current = null;
      }

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
      if (speechRecognitionRef.current) {
        try {
          speechRecognitionRef.current.stop();
        } catch {
          // Ignore browser speech recognition stop errors.
        }
      }
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
    }
  };

  const handleAudioStop = async () => {
    const optimisticVoiceQuery = voiceDraftTranscriptRef.current.trim();
    const insertedOptimisticUser = Boolean(optimisticVoiceQuery);

    setIsLoading(true);
    setIsStreamingResponse(true);
    setIsWaitingForFirstToken(true);
    setHasStartedChat(true);
    setMessages((prev) => {
      const next = [...prev];
      if (insertedOptimisticUser) {
        next.push({ role: 'user', text: optimisticVoiceQuery, isAudio: true });
      }
      next.push({ role: 'bot', text: '', isAudio: true });
      return next;
    });

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

      setIsWaitingForFirstToken(false);
      if (insertedOptimisticUser) {
        setLatestUserMessageText(userQuery);
      } else {
        insertUserMessageBeforeLatestVoiceBot(userQuery);
      }
      setLatestBotMessageText(botReply);
      void refreshSuggestedQuestions(sessionId, userQuery);

      const audioResponseBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioResponseBlob);
      const audio = new Audio(audioUrl);
      await audio.play();
      audio.onended = () => URL.revokeObjectURL(audioUrl);
    } catch {
      setIsWaitingForFirstToken(false);
      setLatestBotMessageText('Sorry, failed to process audio.');
    } finally {
      voiceDraftTranscriptRef.current = '';
      setIsWaitingForFirstToken(false);
      setIsStreamingResponse(false);
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
                  className={`p-3 rounded-2xl text-sm max-w-[94%] sm:max-w-[92%] shadow-sm bg-[var(--vtl-panel)] border border-[var(--vtl-border)] text-[var(--vtl-text)] ${
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
                          const label = getCitationLabel(citation, citationIndex);
                          if (citation.url) {
                            return (
                              <li key={`${label}-${citationIndex}`}>
                                <a
                                  href={citation.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-[var(--vtl-primary)] underline break-words"
                                  title={citation.url}
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

                  {msg.role === 'bot' && Array.isArray(msg.videos) && msg.videos.length > 0 && (
                    <div className="mt-3 border-t border-[var(--vtl-border)] pt-2">
                      <div className="text-xs font-semibold text-[var(--vtl-muted)] mb-2">Videos</div>
                      <div className="space-y-3">
                        {msg.videos.map((video: VideoSource, videoIndex: number) => (
                          <div key={`${video.url}-${videoIndex}`} className="rounded-lg border border-[var(--vtl-border)] overflow-hidden bg-white">
                            <iframe
                              src={video.embedUrl}
                              title={video.title || `Video ${videoIndex + 1}`}
                              className="w-full h-44 sm:h-52 border-0"
                              loading="lazy"
                              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                              allowFullScreen
                            />
                            <a
                              href={video.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="block px-2 py-1 text-xs text-[var(--vtl-primary)] underline break-all"
                            >
                              {video.url}
                            </a>
                          </div>
                        ))}
                      </div>
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
