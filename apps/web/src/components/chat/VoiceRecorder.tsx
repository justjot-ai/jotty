/**
 * VoiceRecorder - Voice Recording UI
 * ==================================
 *
 * Voice recording interface using useVoice() hook.
 * Shows recording state, waveform animation, and transcription status.
 */

'use client';

import { useEffect } from 'react';
import { useVoice } from '@/lib/jotty/hooks';
import { Mic, MicOff, X } from 'lucide-react';

interface VoiceRecorderProps {
  onTranscript: (text: string) => void;
  onCancel: () => void;
}

export default function VoiceRecorder({ onTranscript, onCancel }: VoiceRecorderProps) {
  const voice = useVoice();

  useEffect(() => {
    // Auto-start recording when component mounts
    voice.startRecording();
  }, []);

  const handleStop = async () => {
    const transcript = await voice.stopRecording();
    if (transcript) {
      onTranscript(transcript);
    } else if (voice.error) {
      // Error already displayed, just close
      onCancel();
    }
  };

  return (
    <div className="flex flex-col items-center gap-4 p-6 bg-gray-800">
      {/* Recording Indicator */}
      <div className="relative">
        {/* Pulsing ring animation */}
        {voice.recording && (
          <>
            <div className="absolute inset-0 rounded-full bg-red-500 animate-ping opacity-75" />
            <div className="absolute inset-0 rounded-full bg-red-500 animate-pulse" />
          </>
        )}

        {/* Microphone Button */}
        <button
          onClick={handleStop}
          disabled={voice.transcribing}
          className="relative z-10 p-8 bg-red-600 hover:bg-red-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-full transition-colors"
        >
          <Mic size={32} />
        </button>
      </div>

      {/* Status Text */}
      <div className="text-center">
        {voice.recording && (
          <div className="text-lg font-semibold text-red-400 mb-1">
            Recording...
          </div>
        )}
        {voice.transcribing && (
          <div className="text-lg font-semibold text-emerald-400 mb-1">
            Transcribing...
          </div>
        )}
        <div className="text-sm text-gray-400">
          {voice.recording ? 'Tap to stop recording' : 'Processing audio...'}
        </div>
      </div>

      {/* Waveform Animation (simple bars) */}
      {voice.recording && (
        <div className="flex items-center gap-1 h-12">
          {Array.from({ length: 20 }).map((_, i) => (
            <div
              key={i}
              className="w-1 bg-red-400 rounded-full animate-pulse"
              style={{
                height: `${20 + Math.random() * 60}%`,
                animationDelay: `${i * 50}ms`,
                animationDuration: `${600 + Math.random() * 400}ms`,
              }}
            />
          ))}
        </div>
      )}

      {/* Error Display */}
      {voice.error && (
        <div className="w-full max-w-md p-3 bg-red-900/50 border border-red-700 rounded-lg text-sm">
          <strong>Error:</strong> {voice.error}
        </div>
      )}

      {/* Cancel Button */}
      <button
        onClick={onCancel}
        disabled={voice.transcribing}
        className="flex items-center gap-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 disabled:bg-gray-800 disabled:cursor-not-allowed rounded-lg transition-colors"
      >
        <X size={16} />
        Cancel
      </button>

      {/* Instructions */}
      <div className="text-xs text-gray-500 text-center max-w-md">
        {voice.recording ? (
          <>
            Speak clearly into your microphone. Tap the red button when finished.
          </>
        ) : (
          <>
            Your audio is being transcribed. This may take a few seconds.
          </>
        )}
      </div>
    </div>
  );
}
