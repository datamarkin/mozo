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

/**
 * The file the next run reads, chosen on a node whose parameter is a `source`.
 *
 * A store rather than a prop, because the node that offers the picker and the code that submits
 * the run are at opposite ends of the tree. One file per run, which is what the server accepts:
 * the upload binds to whichever parameter declares a `source`, and a workflow with two of those is
 * refused rather than guessed at.
 *
 * An image or a video, and this does not know which. Deciding is the server's -- one input node
 * reads either, from the extension -- so a picker that filtered by kind here would be re-deciding
 * it, and wrongly: it filtered to `image/*`, which is what made a video unselectable.
 */
export const chosenFile = writable(null);
