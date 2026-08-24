import { writable } from 'svelte/store';

// Sidebar drawer state management
export const sidebarMode = writable('closed'); // 'palette' | 'properties' | 'closed'
export const isSidebarOpen = writable(false);

// Function to open sidebar with specific mode
export function openSidebar(mode) {
    sidebarMode.set(mode);
    isSidebarOpen.set(true);
}

// Function to close sidebar
export function closeSidebar() {
    isSidebarOpen.set(false);
    // Keep mode for potential reopening
}

// Function to toggle sidebar mode (useful for switching between palette and properties)
export function toggleSidebarMode(mode) {
    sidebarMode.update(currentMode => {
        if (currentMode === mode) {
            closeSidebar();
            return 'closed';
        } else {
            openSidebar(mode);
            return mode;
        }
    });
}

// Pending connection state management
export const pendingConnection = writable(null);

// Function to set pending connection when handle is clicked
export function setPendingConnection(nodeId, handleId, handleType) {
    pendingConnection.set({ nodeId, handleId, handleType });
}

// Function to clear pending connection
export function clearPendingConnection() {
    pendingConnection.set(null);
}

// App config from window (injected by Jinja2)
function getAppConfig() {
    if (typeof window !== 'undefined' && window.APP_CONFIG) {
        return window.APP_CONFIG;
    }
    return { deploymentMode: 'local', user: null, workflowId: null };
}

export const appConfig = writable(getAppConfig());
