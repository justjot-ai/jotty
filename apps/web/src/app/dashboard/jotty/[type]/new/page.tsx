'use client';

import { useRouter, useParams } from 'next/navigation';
import { EntityForm } from '@/jotty/components/management/EntityForm';
import { JottyEntityType } from '@/jotty/types';

export default function NewJottyEntityPage() {
  const router = useRouter();
  const params = useParams();
  const type = (params.type || 'agent') as JottyEntityType;

  const handleSave = async (data: any) => {
    const res = await fetch('/api/jotty/entities', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        type,
        ...data,
      }),
    });

    if (res.ok) {
      const result = await res.json();
      router.push(`/dashboard/jotty/${type}/${result.entity.id}`);
    } else {
      throw new Error('Failed to create entity');
    }
  };

  const handleCancel = () => {
    router.push('/dashboard/jotty');
  };

  return (
    <div className="min-h-screen bg-surface-2 p-4 md:p-6">
      <div className="max-w-3xl mx-auto">
        <EntityForm
          entityType={type}
          onSave={handleSave}
          onCancel={handleCancel}
        />
      </div>
    </div>
  );
}
