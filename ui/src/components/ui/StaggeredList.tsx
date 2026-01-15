'use client';

import React, { useEffect, useRef, useState } from 'react';
import { cn } from '@/lib/utils';

interface StaggeredListProps {
  children: React.ReactNode[];
  className?: string;
  delay?: number;
  stagger?: number;
  animation?: 'fade-up' | 'fade-in' | 'scale-in';
}

export function StaggeredList({
  children,
  className,
  delay = 0,
  stagger = 50,
  animation = 'fade-up',
}: StaggeredListProps) {
  const [isVisible, setIsVisible] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.disconnect();
        }
      },
      {
        threshold: 0.1,
        rootMargin: '50px',
      }
    );

    if (containerRef.current) {
      observer.observe(containerRef.current);
    }

    return () => observer.disconnect();
  }, []);

  const animationClasses = {
    'fade-up': 'translate-y-4 opacity-0',
    'fade-in': 'opacity-0',
    'scale-in': 'scale-95 opacity-0',
  };

  const activeAnimationClasses = {
    'fade-up': 'translate-y-0 opacity-100',
    'fade-in': 'opacity-100',
    'scale-in': 'scale-100 opacity-100',
  };

  return (
    <div ref={containerRef} className={className}>
      {React.Children.map(children, (child, index) => (
        <div
          className={cn(
            'transition-all duration-500 ease-out',
            isVisible ? activeAnimationClasses[animation] : animationClasses[animation]
          )}
          style={{
            transitionDelay: `${delay + index * stagger}ms`,
          }}
        >
          {child}
        </div>
      ))}
    </div>
  );
}
