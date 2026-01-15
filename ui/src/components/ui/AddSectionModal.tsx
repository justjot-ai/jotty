'use client';

import React, { useState, useMemo, useCallback, useRef, useEffect } from 'react';
import Link from 'next/link';
import { SectionType, SECTION_REGISTRY, SECTION_TYPE_INFO } from '@/lib/section-registry';
import { useCommunitySections, getCommunitySectionType } from '@/lib/community/loader';

// Get unique categories from registry
const CATEGORIES = ['All', ...Array.from(new Set(SECTION_REGISTRY.map(s => s.category)))];

// Quick start types - most commonly used
const QUICK_START_TYPES = ['text', 'todos', 'code', 'mermaid', 'csv', 'kanban-board'];

export interface AddSectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  onAdd: (title: string, type: SectionType) => Promise<void>;
  isLoading?: boolean;
}

export function AddSectionModal({ isOpen, onClose, onAdd, isLoading = false }: AddSectionModalProps) {
  // Form state
  const [title, setTitle] = useState('');
  const [selectedType, setSelectedType] = useState<SectionType>('text');
  const [searchQuery, setSearchQuery] = useState('');
  const [activeCategory, setActiveCategory] = useState('All');
  const [showCommunity, setShowCommunity] = useState(false);

  // Refs
  const searchInputRef = useRef<HTMLInputElement>(null);
  const titleInputRef = useRef<HTMLInputElement>(null);
  const categoryScrollRef = useRef<HTMLDivElement>(null);

  // Community sections
  const { sections: communitySections, loading: communityLoading } = useCommunitySections();

  // Reset form when modal opens
  useEffect(() => {
    if (isOpen) {
      setTitle('');
      setSelectedType('text');
      setSearchQuery('');
      setActiveCategory('All');
      setShowCommunity(false);
      // Focus search input on desktop, title on mobile for faster flow
      setTimeout(() => {
        if (window.innerWidth >= 768) {
          searchInputRef.current?.focus();
        }
      }, 100);
    }
  }, [isOpen]);

  // Filter section types based on search and category
  const filteredTypes = useMemo(() => {
    const types = [...SECTION_REGISTRY];

    // Filter by search
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      return types.filter(t =>
        t.label.toLowerCase().includes(query) ||
        t.description.toLowerCase().includes(query) ||
        t.category.toLowerCase().includes(query)
      );
    }

    // Filter by category
    if (activeCategory !== 'All') {
      return types.filter(t => t.category === activeCategory);
    }

    return types;
  }, [searchQuery, activeCategory]);

  // Get quick start types info
  const quickStartTypes = useMemo(() => {
    return QUICK_START_TYPES.map(value => {
      const type = SECTION_REGISTRY.find(t => t.value === value);
      return type ? { value: type.value as SectionType, icon: type.icon, label: type.label } : null;
    }).filter(Boolean) as { value: SectionType; icon: string; label: string }[];
  }, []);

  // Handle type selection
  const handleSelectType = useCallback((type: SectionType) => {
    setSelectedType(type);
    // Focus title input after selection
    setTimeout(() => titleInputRef.current?.focus(), 50);
  }, []);

  // Handle submit
  const handleSubmit = useCallback(async () => {
    if (!title.trim() || isLoading) return;
    await onAdd(title.trim(), selectedType);
  }, [title, selectedType, isLoading, onAdd]);

  // Handle key press in title input
  const handleTitleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && title.trim() && !isLoading) {
      e.preventDefault();
      handleSubmit();
    }
  }, [title, isLoading, handleSubmit]);

  // Handle backdrop click
  const handleBackdropClick = useCallback((e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  }, [onClose]);

  // Handle escape key
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };
    window.addEventListener('keydown', handleEscape);
    return () => window.removeEventListener('keydown', handleEscape);
  }, [isOpen, onClose]);

  // Get selected type info
  const selectedTypeInfo = useMemo(() => {
    const type = SECTION_REGISTRY.find(t => t.value === selectedType);
    if (type) return { icon: type.icon, label: type.label, description: type.description };

    // Check community sections
    const communitySection = communitySections.find(s => getCommunitySectionType(s.definition.id) === selectedType);
    if (communitySection) {
      return {
        icon: communitySection.definition.icon,
        label: communitySection.definition.name,
        description: communitySection.definition.description || ''
      };
    }

    return { icon: '📝', label: 'Text', description: '' };
  }, [selectedType, communitySections]);

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-end sm:items-center justify-center bg-black/50 p-0 sm:p-4"
      onClick={handleBackdropClick}
    >
      <div className="bg-white w-full sm:rounded-xl shadow-2xl sm:max-w-2xl max-h-[95vh] sm:max-h-[85vh] flex flex-col overflow-hidden rounded-t-xl sm:rounded-xl">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-gray-200 bg-gray-50 shrink-0">
          <h2 className="text-lg font-semibold text-gray-900">Add Section</h2>
          <button
            onClick={onClose}
            className="p-2 -mr-2 text-gray-500 hover:text-gray-700 hover:bg-gray-100 rounded-lg transition-colors"
            aria-label="Close"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto min-h-0">
          <div className="p-4 space-y-4">
            {/* Search Input */}
            <div className="relative">
              <svg
                className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input
                ref={searchInputRef}
                type="text"
                value={searchQuery}
                onChange={(e) => {
                  setSearchQuery(e.target.value);
                  if (e.target.value) setActiveCategory('All'); // Reset category when searching
                }}
                placeholder="Search section types..."
                className="w-full pl-10 pr-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-900 placeholder-gray-500"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-3 top-1/2 -translate-y-1/2 p-1 text-gray-400 hover:text-gray-600"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              )}
            </div>

            {/* Quick Start Section - Only show when not searching */}
            {!searchQuery && activeCategory === 'All' && (
              <div>
                <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">Quick Start</h3>
                <div className="grid grid-cols-3 sm:grid-cols-6 gap-2">
                  {quickStartTypes.map((type) => (
                    <button
                      key={type.value}
                      onClick={() => handleSelectType(type.value)}
                      className={`flex flex-col items-center gap-1 p-3 rounded-lg border-2 transition-all ${
                        selectedType === type.value
                          ? 'border-blue-500 bg-blue-50 text-blue-700 shadow-sm'
                          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50 text-gray-700'
                      }`}
                    >
                      <span className="text-2xl">{type.icon}</span>
                      <span className="text-xs font-medium text-center leading-tight">{type.label.split(' /')[0].split(' ')[0]}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Category Tabs */}
            <div
              ref={categoryScrollRef}
              className="flex gap-1 overflow-x-auto pb-1 -mx-4 px-4 scrollbar-hide"
              style={{ scrollbarWidth: 'none', msOverflowStyle: 'none' }}
            >
              {CATEGORIES.map((category) => {
                const count = category === 'All'
                  ? SECTION_REGISTRY.length
                  : SECTION_REGISTRY.filter(t => t.category === category).length;
                return (
                  <button
                    key={category}
                    onClick={() => {
                      setActiveCategory(category);
                      setSearchQuery('');
                    }}
                    className={`flex-shrink-0 px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                      activeCategory === category
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {category}
                    <span className={`ml-1 text-xs ${activeCategory === category ? 'text-blue-200' : 'text-gray-500'}`}>
                      {count}
                    </span>
                  </button>
                );
              })}
              {/* Community Tab */}
              {communitySections.length > 0 && (
                <button
                  onClick={() => setShowCommunity(!showCommunity)}
                  className={`flex-shrink-0 px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                    showCommunity
                      ? 'bg-purple-600 text-white'
                      : 'bg-purple-100 text-purple-700 hover:bg-purple-200'
                  }`}
                >
                  Community
                  <span className={`ml-1 text-xs ${showCommunity ? 'text-purple-200' : 'text-purple-500'}`}>
                    {communitySections.length}
                  </span>
                </button>
              )}
            </div>

            {/* Section Type Grid */}
            {!showCommunity && (
              <div>
                <div className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2 flex items-center justify-between">
                  <span>
                    {searchQuery
                      ? `Results (${filteredTypes.length})`
                      : activeCategory === 'All'
                        ? 'All Section Types'
                        : activeCategory
                    }
                  </span>
                  {selectedType && SECTION_REGISTRY.find(t => t.value === selectedType) && (
                    <span className="normal-case text-blue-600 font-normal flex items-center gap-1">
                      <span>{SECTION_TYPE_INFO[selectedType]?.icon}</span>
                      <span>{SECTION_TYPE_INFO[selectedType]?.label}</span>
                    </span>
                  )}
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-[240px] overflow-y-auto pr-1">
                  {filteredTypes.map((type) => (
                    <button
                      key={type.value}
                      onClick={() => handleSelectType(type.value as SectionType)}
                      className={`flex items-center gap-2 p-2.5 rounded-lg border text-left transition-all group ${
                        selectedType === type.value
                          ? 'border-blue-500 bg-blue-50 text-blue-700 shadow-sm'
                          : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50 text-gray-700'
                      }`}
                    >
                      <span className="text-lg shrink-0">{type.icon}</span>
                      <div className="min-w-0 flex-1">
                        <span className="text-sm font-medium block truncate">{type.label}</span>
                        <span className={`text-xs block truncate ${
                          selectedType === type.value ? 'text-blue-500' : 'text-gray-500'
                        }`}>
                          {type.description}
                        </span>
                      </div>
                    </button>
                  ))}
                  {filteredTypes.length === 0 && (
                    <div className="col-span-full text-center py-8 text-gray-500">
                      <p className="text-sm">No section types found</p>
                      <button
                        onClick={() => { setSearchQuery(''); setActiveCategory('All'); }}
                        className="text-blue-600 hover:underline text-sm mt-1"
                      >
                        Clear filters
                      </button>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Community Sections */}
            {showCommunity && (
              <div>
                <div className="text-xs font-medium text-purple-600 uppercase tracking-wide mb-2">
                  Installed Community Sections
                </div>
                {communityLoading ? (
                  <div className="text-center py-4 text-gray-500">Loading...</div>
                ) : communitySections.length > 0 ? (
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-[200px] overflow-y-auto">
                    {communitySections.map((section) => {
                      const communityType = getCommunitySectionType(section.definition.id);
                      return (
                        <button
                          key={section.definition.id}
                          onClick={() => handleSelectType(communityType as SectionType)}
                          className={`flex items-center gap-2 p-2.5 rounded-lg border text-left transition-all ${
                            selectedType === communityType
                              ? 'border-purple-500 bg-purple-50 text-purple-700 shadow-sm'
                              : 'border-purple-200 hover:border-purple-300 hover:bg-purple-50 text-gray-700'
                          }`}
                        >
                          <span className="text-lg shrink-0">{section.definition.icon}</span>
                          <div className="min-w-0 flex-1">
                            <span className="text-sm font-medium block truncate">{section.definition.name}</span>
                            {section.definition.description && (
                              <span className={`text-xs block truncate ${
                                selectedType === communityType ? 'text-purple-500' : 'text-gray-500'
                              }`}>
                                {section.definition.description}
                              </span>
                            )}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                ) : (
                  <div className="p-4 bg-purple-50 rounded-lg border border-purple-200 text-center">
                    <p className="text-sm text-purple-700 mb-2">No community sections installed</p>
                    <Link
                      href="/dashboard/community"
                      className="text-sm text-purple-600 hover:text-purple-800 font-medium hover:underline"
                    >
                      Browse Community Sections →
                    </Link>
                  </div>
                )}
              </div>
            )}

            {/* Marketplace Link when no community sections */}
            {!showCommunity && communitySections.length === 0 && !communityLoading && (
              <div className="p-3 bg-purple-50 rounded-lg border border-purple-200">
                <p className="text-sm text-purple-700 mb-1">Want more section types?</p>
                <Link href="/dashboard/community" className="text-sm text-purple-600 hover:underline font-medium">
                  Browse Community Sections →
                </Link>
              </div>
            )}
          </div>
        </div>

        {/* Footer - Fixed at bottom */}
        <div className="border-t border-gray-200 bg-gray-50 p-4 shrink-0">
          {/* Selected Type Preview */}
          <div className="flex items-center gap-2 mb-3 px-1">
            <span className="text-lg">{selectedTypeInfo.icon}</span>
            <span className="text-sm font-medium text-gray-700">{selectedTypeInfo.label}</span>
            {selectedTypeInfo.description && (
              <span className="text-xs text-gray-500 hidden sm:inline">— {selectedTypeInfo.description}</span>
            )}
          </div>

          {/* Title Input and Actions */}
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="flex-1">
              <input
                ref={titleInputRef}
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                onKeyDown={handleTitleKeyDown}
                placeholder="Enter section title..."
                className="w-full px-4 py-2.5 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-gray-900 placeholder-gray-500"
              />
            </div>
            <div className="flex gap-2 sm:gap-3">
              <button
                onClick={onClose}
                className="flex-1 sm:flex-none px-4 py-2.5 border border-gray-300 text-gray-700 font-medium rounded-lg hover:bg-gray-50 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handleSubmit}
                disabled={!title.trim() || isLoading}
                className="flex-1 sm:flex-none px-6 py-2.5 bg-blue-600 hover:bg-blue-700 text-white font-medium rounded-lg disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2 min-w-[120px]"
              >
                {isLoading && (
                  <svg className="w-4 h-4 animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                )}
                {isLoading ? 'Adding...' : 'Add Section'}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
