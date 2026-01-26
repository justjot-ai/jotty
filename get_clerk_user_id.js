// Run this in browser console on https://justjot.ai/dashboard
// Copy and paste into browser console (F12 > Console tab)

console.log("🔍 Getting Clerk User ID...");
console.log();

// Method 1: Direct Clerk object
if (window.Clerk?.user?.id) {
    console.log("✅ Found Clerk User ID:");
    console.log("   " + window.Clerk.user.id);
    console.log();
    console.log("Copy this: " + window.Clerk.user.id);
} else if (window.__clerk?.user?.id) {
    console.log("✅ Found Clerk User ID:");
    console.log("   " + window.__clerk.user.id);
    console.log();
    console.log("Copy this: " + window.__clerk.user.id);
} else {
    console.log("❌ Clerk user ID not found");
    console.log();
    console.log("Try:");
    console.log("   1. Make sure you're logged in");
    console.log("   2. Refresh the page");
    console.log("   3. Check Network tab for API calls");
}

// Method 2: Check localStorage
console.log();
console.log("📦 Checking localStorage...");
const clerkSession = localStorage.getItem('clerk-session');
if (clerkSession) {
    try {
        const session = JSON.parse(clerkSession);
        if (session?.userId) {
            console.log("✅ Found in localStorage:");
            console.log("   " + session.userId);
        }
    } catch (e) {
        console.log("   Could not parse localStorage");
    }
}

// Method 3: Check an existing idea's userId
console.log();
console.log("💡 To check an existing idea's userId:");
console.log("   1. Open an idea in your dashboard");
console.log("   2. Open Network tab");
console.log("   3. Look for API call to /api/ideas/{id}");
console.log("   4. Check the response for 'userId' field");
