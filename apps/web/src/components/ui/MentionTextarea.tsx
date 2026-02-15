'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';

interface User {
  authId: string;
  email: string;
  firstName: string;
  lastName: string;
  fullName: string;
  imageUrl?: string;
}

interface MentionTextareaProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit?: () => void;
  placeholder?: string;
  rows?: number;
  className?: string;
  id?: string;
}

interface AutocompletePosition {
  top: number;
  left: number;
}

const MentionTextarea: React.FC<MentionTextareaProps> = ({
  value,
  onChange,
  onSubmit,
  placeholder = 'Type @ to mention someone...',
  rows = 2,
  className = '',
  id,
}) => {
  const [users, setUsers] = useState<User[]>([]);
  const [showAutocomplete, setShowAutocomplete] = useState(false);
  const [filteredUsers, setFilteredUsers] = useState<User[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [mentionQuery, setMentionQuery] = useState('');
  const [autocompletePosition, setAutocompletePosition] = useState<AutocompletePosition>({ top: 0, left: 0 });
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const autocompleteRef = useRef<HTMLDivElement>(null);

  // Fetch users on mount
  useEffect(() => {
    const fetchUsers = async () => {
      try {
        const response = await fetch('/api/users');
        if (response.ok) {
          const data = await response.json();
          setUsers(data.users || []);
        }
      } catch (error) {
        console.error('Failed to fetch users:', error);
      }
    };
    fetchUsers();
  }, []);

  // Calculate autocomplete position based on cursor
  const calculateAutocompletePosition = useCallback(() => {
    if (!textareaRef.current) return;

    const textarea = textareaRef.current;
    const cursorPosition = textarea.selectionStart;

    // Create a mirror div to calculate cursor position
    const mirror = document.createElement('div');
    const computedStyle = window.getComputedStyle(textarea);

    // Copy styles from textarea
    mirror.style.position = 'absolute';
    mirror.style.visibility = 'hidden';
    mirror.style.whiteSpace = 'pre-wrap';
    mirror.style.wordWrap = 'break-word';
    mirror.style.font = computedStyle.font;
    mirror.style.padding = computedStyle.padding;
    mirror.style.border = computedStyle.border;
    mirror.style.width = `${textarea.offsetWidth}px`;

    // Add text up to cursor
    mirror.textContent = value.substring(0, cursorPosition);
    document.body.appendChild(mirror);

    // Get dimensions
    const mirrorRect = mirror.getBoundingClientRect();
    const textareaRect = textarea.getBoundingClientRect();

    document.body.removeChild(mirror);

    // Calculate position (approximate)
    const lineHeight = parseInt(computedStyle.lineHeight);
    const lines = mirror.textContent.split('\n').length;
    const top = textareaRect.bottom + 4; // Position below textarea
    const left = textareaRect.left;

    setAutocompletePosition({ top, left });
  }, [value]);

  // Handle text change
  const handleTextChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    const newValue = e.target.value;
    onChange(newValue);

    const cursorPosition = e.target.selectionStart;
    const textBeforeCursor = newValue.substring(0, cursorPosition);

    // Find the last @ symbol
    const lastAtIndex = textBeforeCursor.lastIndexOf('@');

    if (lastAtIndex !== -1) {
      // Check if @ is at the start or preceded by whitespace
      const charBeforeAt = lastAtIndex > 0 ? textBeforeCursor[lastAtIndex - 1] : ' ';
      const isValidMention = charBeforeAt === ' ' || charBeforeAt === '\n' || lastAtIndex === 0;

      if (isValidMention) {
        const query = textBeforeCursor.substring(lastAtIndex + 1);

        // Check if there's a space after @ (which would end the mention)
        if (!query.includes(' ') && !query.includes('\n')) {
          setMentionQuery(query);

          // Filter users based on query
          const filtered = users.filter(user =>
            user.fullName.toLowerCase().includes(query.toLowerCase()) ||
            user.email.toLowerCase().includes(query.toLowerCase()) ||
            user.firstName.toLowerCase().includes(query.toLowerCase()) ||
            user.lastName.toLowerCase().includes(query.toLowerCase())
          ).slice(0, 5); // Limit to 5 results

          setFilteredUsers(filtered);
          setSelectedIndex(0);
          setShowAutocomplete(filtered.length > 0);
          calculateAutocompletePosition();
          return;
        }
      }
    }

    setShowAutocomplete(false);
  };

  // Insert mention
  const insertMention = useCallback((user: User) => {
    if (!textareaRef.current) return;

    const cursorPosition = textareaRef.current.selectionStart;
    const textBeforeCursor = value.substring(0, cursorPosition);
    const textAfterCursor = value.substring(cursorPosition);

    // Find the @ position
    const lastAtIndex = textBeforeCursor.lastIndexOf('@');

    if (lastAtIndex !== -1) {
      const beforeMention = value.substring(0, lastAtIndex);
      const mention = `@${user.fullName}`;
      const newValue = beforeMention + mention + ' ' + textAfterCursor;

      onChange(newValue);

      // Set cursor position after the mention
      setTimeout(() => {
        if (textareaRef.current) {
          const newCursorPos = beforeMention.length + mention.length + 1;
          textareaRef.current.selectionStart = newCursorPos;
          textareaRef.current.selectionEnd = newCursorPos;
          textareaRef.current.focus();
        }
      }, 0);
    }

    setShowAutocomplete(false);
  }, [value, onChange]);

  // Handle keyboard navigation
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (showAutocomplete) {
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex(prev => (prev + 1) % filteredUsers.length);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex(prev => (prev - 1 + filteredUsers.length) % filteredUsers.length);
      } else if (e.key === 'Enter' && !e.shiftKey && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        if (filteredUsers[selectedIndex]) {
          insertMention(filteredUsers[selectedIndex]);
        }
      } else if (e.key === 'Escape') {
        e.preventDefault();
        setShowAutocomplete(false);
      }
    } else {
      // Handle submit with Cmd/Ctrl + Enter
      if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        onSubmit?.();
      }
    }
  };

  // Close autocomplete when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (
        autocompleteRef.current &&
        !autocompleteRef.current.contains(event.target as Node) &&
        textareaRef.current &&
        !textareaRef.current.contains(event.target as Node)
      ) {
        setShowAutocomplete(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Scroll selected item into view
  useEffect(() => {
    if (showAutocomplete && autocompleteRef.current) {
      const selectedElement = autocompleteRef.current.children[selectedIndex] as HTMLElement;
      if (selectedElement) {
        selectedElement.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
      }
    }
  }, [selectedIndex, showAutocomplete]);

  return (
    <div className="relative">
      <textarea
        ref={textareaRef}
        id={id}
        value={value}
        onChange={handleTextChange}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        rows={rows}
        className={className}
      />

      {showAutocomplete && filteredUsers.length > 0 && (
        <div
          ref={autocompleteRef}
          className="fixed z-50 bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg max-h-60 overflow-y-auto"
          style={{
            top: `${autocompletePosition.top}px`,
            left: `${autocompletePosition.left}px`,
            minWidth: '250px',
            maxWidth: '350px',
          }}
        >
          {filteredUsers.map((user, index) => (
            <button
              key={user.authId}
              onClick={() => insertMention(user)}
              className={`w-full text-left flex items-center touch-manipulation transition-colors
                         px-4 py-3 gap-3 sm:px-3 sm:py-2 sm:gap-2
                         hover:bg-gray-100 dark:hover:bg-gray-800
                         ${index === selectedIndex ? 'bg-blue-50 dark:bg-blue-900/30' : ''}`}
              type="button"
            >
              {user.imageUrl ? (
                <img
                  src={user.imageUrl}
                  alt={user.fullName}
                  className="w-10 h-10 sm:w-8 sm:h-8 rounded-full object-cover flex-shrink-0"
                />
              ) : (
                <div className="w-10 h-10 sm:w-8 sm:h-8 rounded-full bg-blue-500 text-white flex items-center justify-center text-sm sm:text-xs font-medium flex-shrink-0">
                  {user.fullName.charAt(0).toUpperCase()}
                </div>
              )}
              <div className="flex-1 min-w-0">
                <div className="text-base sm:text-sm font-medium text-gray-900 dark:text-white truncate">{user.fullName}</div>
                <div className="text-sm sm:text-xs text-gray-500 dark:text-gray-400 truncate">{user.email}</div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default MentionTextarea;
