<script>
    import { sidebarMode, isSidebarOpen, closeSidebar, clearPendingConnection } from './stores.js';
    import PropertiesPanel from './PropertiesPanel.svelte';

    export let availableNodes;
    export let selectedNode;
    export let updateNodeParameters;
    export let executionResults;

    // Prevent event bubbling when clicking inside the drawer
    function handleDrawerClick(event) {
        event.stopPropagation();
    }

    // Handle escape key to close drawer
    function handleKeydown(event) {
        if (event.key === 'Escape' && $isSidebarOpen) {
            closeSidebar();
            clearPendingConnection();
        }
    }
</script>

<svelte:window on:keydown={handleKeydown} />

<!-- Properties drawer sidebar -->
<!-- svelte-ignore a11y-no-noninteractive-element-interactions -->
<div class="drawer-sidebar {$isSidebarOpen ? 'is-open' : ''}" role="dialog" aria-label="Properties panel" tabindex="-1" on:click={handleDrawerClick} on:keydown={handleKeydown}>
    <div class="drawer-content">
        {#if $sidebarMode === 'properties'}
            <PropertiesPanel {selectedNode} {updateNodeParameters} availableNodes={$availableNodes} {executionResults} hideTitle={true} />
        {/if}
    </div>
</div>