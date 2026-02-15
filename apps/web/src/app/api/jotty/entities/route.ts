import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import { connectToDatabase } from '@/lib/mongodb';
import { JottyEntityType } from '@/jotty/types';

// Jotty-specific collections only (no JustJot.ai sections/collections)
const ENTITY_COLLECTIONS: Record<JottyEntityType, string> = {
  'agent': 'customagents',
  'swarm': 'swarmpresets',
  'workflow': 'workflowpresets',
  'tool': 'customagents', // Tools are part of agents
  'mcp-tool': 'customagents', // MCP tools
  'preset': 'swarmpresets', // Swarm presets
  'memory-entity': 'memory_entities', // From Jotty memory system
  'rl-config': 'agent_performance', // RL configurations
};

const ENTITY_NAME_FIELDS: Record<JottyEntityType, string> = {
  'agent': 'name',
  'swarm': 'name',
  'workflow': 'name',
  'tool': 'name',
  'mcp-tool': 'name',
  'preset': 'name',
  'memory-entity': 'entity_name',
  'rl-config': 'agentId',
};

export async function GET(req: NextRequest) {
  try {
    const { userId } = await auth();
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const { searchParams } = new URL(req.url);
    const type = searchParams.get('type') as JottyEntityType;

    if (!type || !ENTITY_COLLECTIONS[type]) {
      return NextResponse.json(
        { error: 'Invalid Jotty entity type' },
        { status: 400 }
      );
    }

    const { db } = await connectToDatabase();
    const collectionName = ENTITY_COLLECTIONS[type];
    const nameField = ENTITY_NAME_FIELDS[type];

    // Build query based on entity type
    const query: any = {};

    // All Jotty entities are user-specific
    query.userId = userId;

    // Special handling for memory entities
    if (type === 'memory-entity') {
      // Memory entities use entity_id, filter by userId in metadata if available
      // For now, we'll fetch all and filter client-side if needed
    }

    // Fetch entities
    const docs = await db
      .collection(collectionName)
      .find(query)
      .sort({ updatedAt: -1, createdAt: -1 })
      .limit(100)
      .toArray();

    // Transform to unified format
    const entities = docs.map((doc: any) => ({
      id: doc._id?.toString() || doc.id,
      type,
      name: doc[nameField] || doc.name || doc.title || `Untitled ${type}`,
      description: doc.description || doc.summary || '',
      data: doc,
      config: doc.config || doc,
      createdAt: doc.createdAt || doc.created_at || new Date().toISOString(),
      updatedAt: doc.updatedAt || doc.updated_at || new Date().toISOString(),
    }));

    return NextResponse.json({ entities });
  } catch (error) {
    console.error('Error fetching Jotty entities:', error);
    return NextResponse.json(
      { error: 'Failed to fetch entities' },
      { status: 500 }
    );
  }
}

export async function POST(req: NextRequest) {
  try {
    const { userId } = await auth();
    if (!userId) {
      return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
    }

    const body = await req.json();
    const { type, name, description, config } = body;

    if (!type || !name) {
      return NextResponse.json(
        { error: 'type and name are required' },
        { status: 400 }
      );
    }

    if (!ENTITY_COLLECTIONS[type as JottyEntityType]) {
      return NextResponse.json(
        { error: 'Invalid Jotty entity type' },
        { status: 400 }
      );
    }

    const { db } = await connectToDatabase();
    const collectionName = ENTITY_COLLECTIONS[type as JottyEntityType];

    const entity = {
      name,
      description: description || '',
      config: config || {},
      userId,
      isPublic: false,
      version: 1,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    const result = await db.collection(collectionName).insertOne(entity);

    return NextResponse.json({
      success: true,
      entity: { ...entity, id: result.insertedId.toString() },
    });
  } catch (error) {
    console.error('Error creating Jotty entity:', error);
    return NextResponse.json(
      { error: 'Failed to create entity' },
      { status: 500 }
    );
  }
}
