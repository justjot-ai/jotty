'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { Textarea } from '@/components/ui/Textarea';
import { Card, CardHeader, CardTitle, CardContent, CardFooter } from '@/components/ui/Card';
import { JottyEntityType } from '@/jotty/types';

interface EntityFormProps {
  entityType: JottyEntityType;
  entityId?: string;
  onSave: (data: any) => Promise<void>;
  onCancel: () => void;
  initialData?: {
    name: string;
    description?: string;
    config: Record<string, any>;
    isPublic: boolean;
  };
}

export function EntityForm({
  entityType,
  entityId,
  onSave,
  onCancel,
  initialData,
}: EntityFormProps) {
  const [name, setName] = useState(initialData?.name || '');
  const [description, setDescription] = useState(initialData?.description || '');
  const [isPublic, setIsPublic] = useState(initialData?.isPublic || false);
  const [config, setConfig] = useState<Record<string, any>>(
    initialData?.config || getDefaultConfig(entityType)
  );
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!name.trim()) {
      setError('Name is required');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      await onSave({
        name: name.trim(),
        description: description.trim(),
        config,
        isPublic,
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save');
    } finally {
      setSaving(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <Card variant="elevated">
        <CardHeader>
          <CardTitle>
            {entityId ? 'Edit' : 'Create'} {getEntityTypeLabel(entityType)}
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {error && (
            <div className="p-3 bg-error-50 dark:bg-error-900/20 border border-error-200 dark:border-error-800 rounded-lg">
              <p className="text-sm text-error-600 dark:text-error-400">{error}</p>
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Name *
            </label>
            <Input
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={`Enter ${entityType} name`}
              required
              className="min-h-[48px]"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Description
            </label>
            <Textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder={`Describe this ${entityType}`}
              rows={3}
              className="min-h-[100px]"
            />
          </div>

          {/* Entity-specific config fields */}
          {renderConfigFields(entityType, config, setConfig)}

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="isPublic"
              checked={isPublic}
              onChange={(e) => setIsPublic(e.target.checked)}
              className="w-5 h-5 rounded border-gray-300 text-primary-600 focus:ring-primary-500"
            />
            <label htmlFor="isPublic" className="text-sm text-gray-700 dark:text-gray-300">
              Make this {entityType} public (others can use it)
            </label>
          </div>
        </CardContent>
        <CardFooter className="flex gap-3 justify-end">
          <Button
            type="button"
            variant="ghost"
            onClick={onCancel}
            disabled={saving}
            className="min-h-[48px]"
          >
            Cancel
          </Button>
          <Button
            type="submit"
            disabled={saving || !name.trim()}
            className="min-h-[48px]"
          >
            {saving ? 'Saving...' : entityId ? 'Update' : 'Create'}
          </Button>
        </CardFooter>
      </Card>
    </form>
  );
}

function getEntityTypeLabel(type: JottyEntityType): string {
  const labels: Record<JottyEntityType, string> = {
    agent: 'Agent',
    swarm: 'Swarm',
    workflow: 'Workflow',
    tool: 'Tool',
    'mcp-tool': 'MCP Tool',
    preset: 'Preset',
    'memory-entity': 'Memory Entity',
    'rl-config': 'RL Config',
  };
  return labels[type];
}

function getDefaultConfig(type: JottyEntityType): Record<string, any> {
  switch (type) {
    case 'agent':
      return {
        systemPrompt: 'You are a helpful AI assistant.',
        temperature: 0.7,
        tools: [],
        maxSteps: 10,
      };
    case 'swarm':
      return {
        agents: [],
        parallel: false,
      };
    case 'workflow':
      return {
        steps: [],
        triggers: [],
      };
    case 'tool':
    case 'mcp-tool':
      return {
        name: '',
        description: '',
        parameters: [],
      };
    case 'preset':
      return {
        agents: [],
        config: {},
      };
    case 'memory-entity':
      return {
        entity_type: 'agent',
        metadata: {},
      };
    case 'rl-config':
      return {
        learningRate: 0.1,
        discountFactor: 0.9,
        explorationRate: 0.2,
      };
    default:
      return {};
  }
}

function renderConfigFields(
  type: JottyEntityType,
  config: Record<string, any>,
  setConfig: (config: Record<string, any>) => void
): React.ReactNode {
  switch (type) {
    case 'agent':
      return (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              System Prompt
            </label>
            <Textarea
              value={config.systemPrompt || ''}
              onChange={(e) => setConfig({ ...config, systemPrompt: e.target.value })}
              placeholder="Enter system prompt for the agent"
              rows={4}
              className="min-h-[100px] font-mono text-sm"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Temperature: {config.temperature || 0.7}
            </label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.1"
              value={config.temperature || 0.7}
              onChange={(e) => setConfig({ ...config, temperature: parseFloat(e.target.value) })}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>Focused</span>
              <span>Creative</span>
            </div>
          </div>
        </div>
      );
    case 'rl-config':
      return (
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Learning Rate: {config.learningRate || 0.1}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={config.learningRate || 0.1}
              onChange={(e) => setConfig({ ...config, learningRate: parseFloat(e.target.value) })}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Discount Factor: {config.discountFactor || 0.9}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={config.discountFactor || 0.9}
              onChange={(e) => setConfig({ ...config, discountFactor: parseFloat(e.target.value) })}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
          </div>
        </div>
      );
    default:
      return (
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Configuration (JSON)
          </label>
          <Textarea
            value={JSON.stringify(config, null, 2)}
            onChange={(e) => {
              try {
                setConfig(JSON.parse(e.target.value));
              } catch {
                // Invalid JSON, ignore
              }
            }}
            rows={6}
            className="min-h-[150px] font-mono text-sm"
          />
        </div>
      );
  }
}
