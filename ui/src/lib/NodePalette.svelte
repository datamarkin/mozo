<script>
    import CategoryIcon from './CategoryIcon.svelte';

    export let availableNodes;
    export let hideTitle = false;
    export let searchQuery = '';

    function onDragStart(event, nodeType) {
        event.dataTransfer.setData('application/reactflow', nodeType);
        event.dataTransfer.effectAllowed = 'move';
    }

    // Filter nodes based on search query
    $: filteredNodes = Object.entries($availableNodes || {}).filter(([nodeType, nodeInfo]) => {
        if (!searchQuery || searchQuery.trim() === '') return true;

        const query = searchQuery.toLowerCase();
        const name = (nodeInfo.name || nodeType).toLowerCase();
        const description = (nodeInfo.description || '').toLowerCase();
        const type = nodeType.toLowerCase();
        const category = (nodeInfo.category || '').toLowerCase();

        return name.includes(query) || description.includes(query) || type.includes(query) || category.includes(query);
    });

    // Organize nodes by categories using backend data
    function getNodeCategories() {
        const nodes = $availableNodes || {};
        const categories = {};

        // Group nodes by their backend-defined category
        for (const [nodeType, nodeInfo] of Object.entries(nodes)) {
            const category = nodeInfo.category || 'Other';

            if (!categories[category]) {
                categories[category] = {
                    nodes: []
                };
            }

            categories[category].nodes.push(nodeType);
        }

        return categories;
    }

    function getNodeDescription(nodeType) {
        const nodeInfo = $availableNodes?.[nodeType];
        return nodeInfo?.description || 'Process image data';
    }

    let expandedCategories = {};

    function toggleCategory(categoryName) {
        expandedCategories[categoryName] = !expandedCategories[categoryName];
        expandedCategories = {...expandedCategories};
    }

    // Initialize all categories as expanded
    $: if ($availableNodes) {
        const categories = getNodeCategories();
        for (const categoryName of Object.keys(categories)) {
            if (!(categoryName in expandedCategories)) {
                expandedCategories[categoryName] = true;
            }
        }
    }
</script>

{#if !hideTitle}
<div class="block mb-1">
    <strong>What's next?</strong>
</div>
{/if}

<nav class="panel">
    {#each filteredNodes as [nodeType, nodeInfo]}
        <div class="panel-block node-panel-block"
             role="button"
             tabindex="0"
             draggable="true"
             on:dragstart={(event) => onDragStart(event, nodeType)}>
            <span class="panel-icon">
                <CategoryIcon category={nodeInfo.category || 'Other'} size={16} />
            </span>
            <div>
                <p class="is-size-8 has-text-weight-bold">{nodeInfo.name || nodeType}</p>
                <p class="is-size-7 has-text-grey">{getNodeDescription(nodeType)}</p>
            </div>
        </div>
    {/each}
</nav>