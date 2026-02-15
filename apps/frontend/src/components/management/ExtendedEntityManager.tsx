'use client';

import { useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Plus, Edit2, Trash2, Search, ExternalLink } from 'lucide-react';
import { Card, CardContent } from '@/components/ui/Card';
import { JottyEntityType } from '@/jotty/types';

interface ExtendedEntity {
  id: string;
  type: JottyEntityType;
  name: string;
  description?: string;
  data: Record<string, any>;
  createdAt: string;
  updatedAt: string;
}

interface ExtendedEntityManagerProps {
  entityType: JottyEntityType;
  title: string;
  description?: string;
  onCreate?: () => void;
  onEdit?: (entity: ExtendedEntity) => void;
  onDelete?: (entity: ExtendedEntity) => void;
}

export function ExtendedEntityManager({
  entityType,
  title,
  description,
  onCreate,
  onEdit,
  onDelete,
}: ExtendedEntityManagerProps) {
  const [entities, setEntities] = useState<ExtendedEntity[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    fetchEntities();
  }, [entityType]);

  const fetchEntities = async () => {
    try {
      setLoading(true);
      const res = await fetch(`/api/jotty/entities?type=${entityType}`);
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
    async (entity: ExtendedEntity) => {
      if (!confirm(`Are you sure you want to delete "${entity.name}"?`)) {
        return;
      }

      try {
        const res = await fetch(`/api/jotty/entities?id=${entity.id}&type=${entityType}`, {
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
    [entities, onDelete, entityType]
  );

  const filteredEntities = entities.filter(entity => {
    const matchesSearch =
      searchQuery === '' ||
      entity.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      entity.description?.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesSearch;
  });

  const getEntityIcon = (type: JottyEntityType) => {
    switch (type) {
      case 'agent': return '🤖';
      case 'swarm': return '👥';
      case 'workflow': return '⚡';
      case 'tool': return '🔧';
      case 'mcp-tool': return '🔌';
      case 'preset': return '📋';
      case 'memory-entity': return '🧠';
      case 'rl-config': return '📊';
      default: return '📄';
    }
  };

  const getEntityLink = (entity: ExtendedEntity) => {
    switch (entityType) {
      case 'agent':
        return `/dashboard/jotty/agents/${entity.id}`;
      case 'swarm':
        return `/dashboard/jotty/swarms/${entity.id}`;
      case 'workflow':
        return `/dashboard/jotty/workflows/${entity.id}`;
      case 'tool':
      case 'mcp-tool':
        return `/dashboard/jotty/tools/${entity.id}`;
      case 'preset':
        return `/dashboard/jotty/presets/${entity.id}`;
      case 'memory-entity':
        return `/dashboard/jotty/memory/${entity.id}`;
      case 'rl-config':
        return `/dashboard/jotty/rl/${entity.id}`;
      default:
        return `/dashboard/jotty/${entityType}/${entity.id}`;
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">{title}</h2>
          {description ? (
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              {description}
            </p>
          ) : (
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Manage your {title.toLowerCase()}
            </p>
          )}
        </div>
        {onCreate && (
          <Button
            onClick={onCreate}
            className="min-h-[48px] w-full sm:w-auto"
          >
            <Plus className="w-4 h-4 mr-2" />
            Create {title.slice(0, -1)}
          </Button>
        )}
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-gray-400" />
        <Input
          type="text"
          placeholder={`Search ${title.toLowerCase()}...`}
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10 min-h-[48px]"
        />
      </div>

      {/* Entity List */}
      {loading ? (
        <div className="text-center py-12">
          <div className="inline-block w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full animate-spin" />
          <p className="mt-4 text-gray-600 dark:text-gray-400">Loading...</p>
        </div>
      ) : filteredEntities.length === 0 ? (
        <Card variant="elevated">
          <CardContent className="p-12 text-center">
            <div className="text-6xl mb-4">{getEntityIcon(entityType)}</div>
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              {searchQuery ? 'No entities found matching your search' : `No ${title.toLowerCase()} yet`}
            </p>
            {!searchQuery && onCreate && (
              <Button onClick={onCreate} variant="outline">
                <Plus className="w-4 h-4 mr-2" />
                Create your first {title.slice(0, -1).toLowerCase()}
              </Button>
            )}
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredEntities.map((entity) => (
            <Card
              key={entity.id}
              variant="elevated"
              hoverable
              className="cursor-pointer"
              onClick={() => {
                const link = getEntityLink(entity);
                if (link) window.location.href = link;
              }}
            >
              <CardContent className="p-4">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-3 flex-1 min-w-0">
                    <div className="text-2xl flex-shrink-0">{getEntityIcon(entityType)}</div>
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
                  </div>
                </div>
                <div className="flex items-center justify-between mt-4">
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(entity.updatedAt).toLocaleDateString()}
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        const link = getEntityLink(entity);
                        if (link) window.open(link, '_blank');
                      }}
                      className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors min-h-[44px] min-w-[44px]"
                      aria-label={`View ${entity.name}`}
                    >
                      <ExternalLink className="w-4 h-4" />
                    </button>
                    {onEdit && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onEdit(entity);
                        }}
                        className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors min-h-[44px] min-w-[44px]"
                        aria-label={`Edit ${entity.name}`}
                      >
                        <Edit2 className="w-4 h-4" />
                      </button>
                    )}
                    {onDelete && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDelete(entity);
                        }}
                        className="p-2 text-error-600 dark:text-error-400 hover:bg-error-50 dark:hover:bg-error-900/20 rounded-lg transition-colors min-h-[44px] min-w-[44px]"
                        aria-label={`Delete ${entity.name}`}
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
