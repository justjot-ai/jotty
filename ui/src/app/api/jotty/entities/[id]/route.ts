import { NextRequest, NextResponse } from 'next/server';
import { auth } from '@clerk/nextjs/server';
import { connectToDatabase } from '@/lib/mongodb';
import { ObjectId } from 'mongodb';
import { JottyEntityType } from '@/jotty/types';

const ENTITY_COLLECTIONS: Record<JottyEntityType, string> = {
  'agent': 'customagents',
  'swarm': 'swarmpresets',
  'workflow': 'workflowpresets',
  'tool': 'customagents',
  'mcp-tool': 'customagents',
  'preset': 'swarmpresets',
  'memory-entity': 'memory_entities',
  'rl-config': 'agent_performance',
};

export async function GET(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
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
    
    const entity = await db.collection(collectionName).findOne({
      _id: new ObjectId(params.id),
      userId,
    });

    if (!entity) {
      return NextResponse.json(
        { error: 'Entity not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({ entity });
  } catch (error) {
    console.error('Error fetching Jotty entity:', error);
    return NextResponse.json(
      { error: 'Failed to fetch entity' },
      { status: 500 }
    );
  }
}

export async function PUT(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
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

    const body = await req.json();
    const { name, description, config } = body;

    const { db } = await connectToDatabase();
    const collectionName = ENTITY_COLLECTIONS[type];
    
    const existingEntity = await db.collection(collectionName).findOne({
      _id: new ObjectId(params.id),
      userId,
    });

    if (!existingEntity) {
      return NextResponse.json(
        { error: 'Entity not found' },
        { status: 404 }
      );
    }

    const update: any = {
      updatedAt: new Date().toISOString(),
    };

    if (name !== undefined) update.name = name;
    if (description !== undefined) update.description = description;
    if (config !== undefined) update.config = config;

    // Increment version if config changed
    if (config !== undefined && JSON.stringify(config) !== JSON.stringify(existingEntity.config)) {
      update.version = (existingEntity.version || 1) + 1;
    }

    await db.collection(collectionName).updateOne(
      { _id: new ObjectId(params.id), userId },
      { $set: update }
    );

    const updatedEntity = await db.collection(collectionName).findOne({
      _id: new ObjectId(params.id),
    });

    return NextResponse.json({
      success: true,
      entity: updatedEntity,
    });
  } catch (error) {
    console.error('Error updating Jotty entity:', error);
    return NextResponse.json(
      { error: 'Failed to update entity' },
      { status: 500 }
    );
  }
}

export async function DELETE(
  req: NextRequest,
  { params }: { params: { id: string } }
) {
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

    const query: any = { _id: new ObjectId(params.id) };
    // All Jotty entities are user-specific
    query.userId = userId;

    const result = await db.collection(collectionName).deleteOne(query);

    if (result.deletedCount === 0) {
      return NextResponse.json(
        { error: 'Entity not found' },
        { status: 404 }
      );
    }

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error deleting Jotty entity:', error);
    return NextResponse.json(
      { error: 'Failed to delete entity' },
      { status: 500 }
    );
  }
}
