/**
 * Jotty Entity Types
 * 
 * Only Jotty-specific entities (no JustJot.ai sections/collections)
 */

export type JottyEntityType = 
  | 'agent'           // Custom AI agents
  | 'swarm'           // Multi-agent swarms
  | 'workflow'        // Agent workflows and presets
  | 'tool'            // Agent tools
  | 'mcp-tool'        // MCP server tools
  | 'preset'          // Agent/swarm presets
  | 'memory-entity'   // Memory graph entities
  | 'rl-config';      // Reinforcement learning configurations

export interface JottyEntity {
  id: string;
  type: JottyEntityType;
  name: string;
  description?: string;
  config?: Record<string, any>;
  data?: Record<string, any>;
  userId: string;
  isPublic?: boolean;
  version?: number;
  createdAt: string;
  updatedAt: string;
}
