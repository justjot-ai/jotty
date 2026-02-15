'use client';

import { useRouter, useParams } from 'next/navigation';
import { useEffect, useState } from 'react';
import { EntityForm } from '@/jotty/components/management/EntityForm';
import { JottyEntityType } from '@/jotty/types';

export default function EditJottyEntityPage() {
  const router = useRouter();
  const params = useParams();
  const type = (params.type || 'agent') as JottyEntityType;
  const id = params.id as string;
  const [initialData, setInitialData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchEntity();
  }, [id, type]);

  const fetchEntity = async () => {
    try {
      const res = await fetch(`/api/jotty/entities/${id}?type=${type}`);
      if (res.ok) {
        const data = await res.json();
        setInitialData(data.entity);
      }
    } catch (error) {
      console.error('Failed to fetch entity:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async (data: any) => {
    const res = await fetch(`/api/jotty/entities/${id}?type=${type}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });

    if (res.ok) {
      router.push(`/dashboard/jotty/${type}/${id}`);
    } else {
      throw new Error('Failed to update entity');
    }
  };

  const handleCancel = () => {
    router.push(`/dashboard/jotty/${type}/${id}`);
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-surface-2 p-4 md:p-6 flex items-center justify-center">
        <div className="text-center">
          <div className="inline-block w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin" />
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-surface-2 p-4 md:p-6">
      <div className="max-w-3xl mx-auto">
        <EntityForm
          entityType={type}
          entityId={id}
          onSave={handleSave}
          onCancel={handleCancel}
          initialData={initialData}
        />
      </div>
    </div>
  );
}
