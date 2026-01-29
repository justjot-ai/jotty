// Test use case for Jotty TypeScript SDK
import { JottyClient, ChatExecuteRequest } from './jotty-sdk';

async function testChat() {
    const client = new JottyClient('http://localhost:8080', 'your-api-key');
    
    const request: ChatExecuteRequest = {
        message: 'Hello, how can you help?',
        history: []
    };
    
    const result = await client.chatExecute(request);
    console.log(`Response: ${result.final_output}`);
    return result;
}

testChat().catch(console.error);
