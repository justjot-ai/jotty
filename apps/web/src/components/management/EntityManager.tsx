'use client';

import { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Plus, Edit2, Trash2, Search, Filter } from 'lucide-react';

// This file is kept for backward compatibility but should use JottyEntityType from @/jotty/types
import { JottyEntityType } from '@/jotty/types';
export type EntityType = JottyEntityType;

interface Entity {
  id: string;
  type: EntityType;
  name: string;
  description?: string;
  config: Record<string, any>;
  userId: string;
  isPublic: boolean;
  version: number;
  createdAt: string;
  updatedAt: string;
}

interface EntityManagerProps {
  entityType: EntityType;
  title: string;
  onCreate?: () => void;
  onEdit?: (entity: Entity) => void;
  onDelete?: (entity: Entity) => void;
  renderEntityCard?: (entity: Entity) => React.ReactNode;
}

export function EntityManager({
  entityType,
  title,
  onCreate,
  onEdit,
  onDelete,
  renderEntityCard,
}: EntityManagerProps) {
  const [entities, setEntities] = useState<Entity[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterPublic, setFilterPublic] = useState<boolean | null>(null);

  useEffect(() => {
    fetchEntities();
  }, [entityType, filterPublic]);

  const fetchEntities = async () => {
    try {
      setLoading(true);
      const params = new URLSearchParams();
      if (filterPublic !== null) {
        params.append('isPublic', filterPublic.toString());
      }
      const res = await fetch(`/api/jotty/entities?type=${entityType}${params.toString() ? '&' + params.toString() : ''}`);
      if (res.ok) {
        const data = await res.json();
        setEntities(data.entities || []);
      }
    } catch (error) {
      console.error('Failed to fetch entities:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = useCallback(
    async (entity: Entity) => {
      if (!confirm(`Are you sure you want to delete "${entity.name}"?`)) {
        return;
      }

      try {
        const res = await fetch(`/api/jotty/entities/${entity.id}`, {
          method: 'DELETE',
        });

        if (res.ok) {
          setEntities(entities.filter(e => e.id !== entity.id));
          onDelete?.(entity);
        }
      } catch (error) {
        console.error('Failed to delete entity:', error);
      }
    },
    [entities, onDelete]
  );

  const filteredEntities = entities.filter(entity => {
    const matchesSearch =
      searchQuery === '' ||
      entity.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entity.description?.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSearch;
  });

  const defaultRenderCard = (entity: Entity) => (
    <div
      key={entity.id}
      className="p-4 bg-white dark:bg-gray-800 rounded-xl shadow-md border border-gray-200 dark:border-gray-700"
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-gray-900 dark:text-white truncate">
            {entity.name}
          </h3>
          {entity.description && (
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1 line-clamp-2">
              {entity.description}
            </p>
          )}
        </div>
        {entity.isPublic && (
          <span className="ml-2 px-2 py-1 text-xs bg-primary-100 text-primary-700 dark:bg-primary-900/30 dark:text-primary-400 rounded-full">
            Public
          </span>
        )}
      </div>
      <div className="flex items-center justify-between mt-4">
        <div className="text-xs text-gray-500 dark:text-gray-400">
          v{entity.version} • {new Date(entity.updatedAt).toLocaleDateString()}
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => onEdit?.(entity)}
            className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors min-h-[44px] min-w-[44px]"
            aria-label={`Edit ${entity.name}`}
          >
            <Edit2 className="w-4 h-4" />
          </button>
          <button
            onClick={() => handleDelete(entity)}
            className="p-2 text-error-600 dark:text-error-400 hover:bg-error-50 dark:hover:bg-error-900/20 rounded-lg transition-colors min-h-[44px] min-w-[44px]"
            aria-label={`Delete ${entity.name}`}
          >
            <Trash2 className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">{title}</h2>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Manage your {entityType}s
          </p>
        </div>
        <Button
          onClick={onCreate}
          className="min-h-[48px] w-full sm:w-auto"
        >
          <Plus className="w-4 h-4 mr-2" />
          Create {title}
        </Button>
      </div>

      {/* Search and Filter */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
          <Input
            type="text"
            placeholder={`Search ${entityType}s...`}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 min-h-[48px]"
          />
        </div>
        <div className="flex gap-2">
          <button
            onClick={() => setFilterPublic(null)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors min-h-[48px] ${
              filterPublic === null
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300'
            }`}
          >
            All
          </button>
          <button
            onClick={() => setFilterPublic(true)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors min-h-[48px] ${
              filterPublic === true
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300'
            }`}
          >
            Public
          </button>
          <button
            onClick={() => setFilterPublic(false)}
            className={`px-4 py-2 rounded-lg font-medium transition-colors min-h-[48px] ${
              filterPublic === false
                ? 'bg-primary-600 text-white'
                : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300'
            }`}
          >
            Private
          </button>
        </div>
      </div>

      {/* Entity List */}
      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin" />
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading...</p>
        </div>
      ) : filteredEntities.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 dark:bg-gray-900 rounded-xl">
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            {searchQuery ? 'No entities found matching your search' : `No ${entityType}s yet`}
          </p>
          {!searchQuery && (
            <Button onClick={onCreate} variant="outline">
              <Plus className="w-4 h-4 mr-2" />
              Create your first {entityType}
            </Button>
          )}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredEntities.map(entity =>
            renderEntityCard ? renderEntityCard(entity) : defaultRenderCard(entity)
          )}
        </div>
      )}
    </div>
  );
}
