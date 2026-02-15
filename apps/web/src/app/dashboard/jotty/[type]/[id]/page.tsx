'use client';

import { useParams, useRouter } from 'next/navigation';
import { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Edit2, Trash2, ArrowLeft } from 'lucide-react';
import { JottyEntityType } from '@/jotty/types';

export default function JottyEntityDetailPage() {
  const params = useParams();
  const router = useRouter();
  const type = params.type as JottyEntityType;
  const id = params.id as string;
  const [entity, setEntity] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchEntity();
  }, [id, type]);

  const fetchEntity = async () => {
    try {
      const res = await fetch(`/api/jotty/entities/${id}?type=${type}`);
      if (res.ok) {
        const data = await res.json();
        setEntity(data.entity);
      }
    } catch (error) {
      console.error('Failed to fetch entity:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this entity?')) {
      return;
    }

    try {
      const res = await fetch(`/api/jotty/entities/${id}?type=${type}`, {
        method: 'DELETE',
      });

      if (res.ok) {
        router.push('/dashboard/jotty');
      }
    } catch (error) {
      console.error('Failed to delete entity:', error);
    }
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

  if (!entity) {
    return (
      <div className="min-h-screen bg-surface-2 p-4 md:p-6">
        <div className="max-w-3xl mx-auto">
          <Card variant="elevated">
            <CardContent className="p-6 text-center">
              <p className="text-gray-600 dark:text-gray-400">Entity not found</p>
              <Button
                onClick={() => router.push('/dashboard/jotty')}
                className="mt-4"
              >
                Back to Jotty Management
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-surface-2 p-4 md:p-6">
      <div className="max-w-3xl mx-auto space-y-6">
        <div className="flex items-center justify-between">
          <Button
            variant="ghost"
            onClick={() => router.push('/dashboard/jotty')}
            className="min-h-[48px]"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </Button>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => router.push(`/dashboard/jotty/${type}/${id}/edit`)}
              className="min-h-[48px]"
            >
              <Edit2 className="w-4 h-4 mr-2" />
              Edit
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              className="min-h-[48px]"
            >
              <Trash2 className="w-4 h-4 mr-2" />
              Delete
            </Button>
          </div>
        </div>

        <Card variant="elevated">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-2xl">{entity.name || entity.entity_name || `Untitled ${type}`}</CardTitle>
              {entity.isPublic && (
                <span className="px-3 py-1 text-sm bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-400 rounded-full">
                  Public
                </span>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {entity.description && (
              <div>
                <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Description
                </h3>
                <p className="text-gray-900 dark:text-white">{entity.description}</p>
              </div>
            )}

            <div>
              <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Configuration
              </h3>
              <pre className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg overflow-x-auto text-sm">
                {JSON.stringify(entity.config || entity, null, 2)}
              </pre>
            </div>

            <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
              {entity.version && <span>Version: {entity.version}</span>}
              {entity.createdAt && (
                <>
                  <span>Created: {new Date(entity.createdAt).toLocaleDateString()}</span>
                  <span>•</span>
                </>
              )}
              {entity.updatedAt && (
                <span>Updated: {new Date(entity.updatedAt).toLocaleDateString()}</span>
              )}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
