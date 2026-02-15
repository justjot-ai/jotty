'use client';

import { useState } from 'react';
import { ExtendedEntityManager } from '@/components/management/ExtendedEntityManager';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/Tabs';
import { Bot, Workflow, Wrench, Users, Brain, TestTube, Settings, Zap } from 'lucide-react';
import { JottyEntityType } from '@/jotty/types';

export default function JottyManagementPage() {
  const [activeTab, setActiveTab] = useState<JottyEntityType>('agent');

  const tabs = [
    { value: 'agent', label: 'Agents', icon: Bot, description: 'AI agents and their configurations' },
    { value: 'swarm', label: 'Swarms', icon: Users, description: 'Multi-agent swarms' },
    { value: 'workflow', label: 'Workflows', icon: Workflow, description: 'Agent workflows and presets' },
    { value: 'tool', label: 'Tools', icon: Wrench, description: 'Agent tools and capabilities' },
    { value: 'mcp-tool', label: 'MCP Tools', icon: Zap, description: 'MCP server tools' },
    { value: 'preset', label: 'Presets', icon: Settings, description: 'Agent and swarm presets' },
    { value: 'memory-entity', label: 'Memory Entities', icon: Brain, description: 'Memory graph entities' },
    { value: 'rl-config', label: 'RL Configs', icon: TestTube, description: 'Reinforcement learning configurations' },
  ];

  const handleCreate = (type: JottyEntityType) => {
    window.location.href = `/dashboard/jotty/${type}/new`;
  };

  const handleEdit = (entity: any) => {
    window.location.href = `/dashboard/jotty/${entity.type}/${entity.id}/edit`;
  };

  return (
    <div className="min-h-screen bg-surface-2 p-4 md:p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-6">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
            Jotty Management
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            Manage agents, swarms, workflows, tools, and all Jotty entities
          </p>
        </div>

        <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as JottyEntityType)}>
          <TabsList className="grid w-full grid-cols-2 md:grid-cols-4 lg:grid-cols-8 mb-6">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <TabsTrigger
                  key={tab.value}
                  value={tab.value}
                  className="flex items-center gap-2 min-h-[48px]"
                >
                  <Icon className="w-4 h-4" />
                  <span className="hidden sm:inline">{tab.label}</span>
                </TabsTrigger>
              );
            })}
          </TabsList>

          {tabs.map((tab) => (
            <TabsContent key={tab.value} value={tab.value}>
              <ExtendedEntityManager
                entityType={tab.value}
                title={tab.label}
                description={tab.description}
                onCreate={() => handleCreate(tab.value)}
                onEdit={handleEdit}
              />
            </TabsContent>
          ))}
        </Tabs>
      </div>
    </div>
  );
}
