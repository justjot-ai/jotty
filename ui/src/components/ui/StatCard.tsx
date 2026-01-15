'use client';

import React from 'react';
import Link from 'next/link';
import { useCountUp } from '@/hooks/useCountUp';

interface StatCardProps {
  title: string;
  value: number;
  icon: React.ReactNode;
  href: string;
  bgColor: string;
  iconColor: string;
  loading?: boolean;
}

export function StatCard({
  title,
  value,
  icon,
  href,
  bgColor,
  iconColor,
  loading = false,
}: StatCardProps) {
  const { count, elementRef } = useCountUp({
    end: value,
    duration: 1200,
    delay: 100,
  });

  return (
    <Link href={href}>
      <div className="bg-surface rounded-lg sm:rounded-md-lg p-2 sm:p-3 md:p-md-4 lg:p-5 shadow-md-2 active:shadow-md-1 transition-all duration-300 hover:shadow-md-3 hover:-translate-y-1 touch-manipulation">
        <div className="flex items-start gap-1.5 sm:gap-2 md:gap-md-3">
          <div className={`w-8 h-8 sm:w-10 sm:h-10 md:w-12 md:h-12 lg:w-14 lg:h-14 ${bgColor} rounded-md sm:rounded-md-md flex items-center justify-center flex-shrink-0 transition-transform duration-300 hover:scale-110`}>
            <div className={`w-4 h-4 sm:w-5 sm:h-5 md:w-6 md:h-6 lg:w-7 lg:h-7 ${iconColor}`}>
              {icon}
            </div>
          </div>
          <div className="min-w-0 flex-1">
            <p className="text-[10px] sm:text-xs md:text-sm lg:text-base font-medium text-gray-600 mb-0.5">
              {title}
            </p>
            {loading ? (
              <div className="h-7 sm:h-8 md:h-9 lg:h-10 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
            ) : (
              <p
                ref={elementRef as any}
                className="text-lg sm:text-xl md:text-2xl lg:text-3xl font-bold text-gray-900 tabular-nums"
              >
                {count}
              </p>
            )}
          </div>
        </div>
      </div>
    </Link>
  );
}
