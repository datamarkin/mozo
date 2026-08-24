<script>
    import { onMount, onDestroy, setContext } from 'svelte';
    import { writable, get } from 'svelte/store';
    import { SvelteFlow, Controls, ControlButton, Background, MiniMap, SvelteFlowProvider } from '@xyflow/svelte';

    import Toolbar from './lib/Toolbar.svelte';
    import CustomNode from './lib/CustomNode.svelte';
    import FlowDropZone from './lib/FlowDropZone.svelte';
    import DrawerSidebar from './lib/DrawerSidebar.svelte';
    import NodePalettePanel from './lib/NodePalettePanel.svelte';
    import { generateNodeClasses } from './lib/utils.js';
    import { fetchCatalogue, fromDocument, stream, toDocument } from './lib/api.js';
    import { openSidebar, closeSidebar, pendingConnection, clearPendingConnection, appConfig } from './lib/stores.js';

    // Every workflow starts here: pixels have to come from somewhere, and this is the node whose
    // path the runner overrides per image.
    const createLoadImageNode = () => ({
        id: 'load_image-1',
        type: 'default',
        position: { x: 100, y: 100 },
        data: { nodeType: 'load_image', parameters: {} },
        class: 'node-category-input'
    });

    let nodes = writable([createLoadImageNode()]);
    let edges = writable([]);
    let selectedNode = writable(null);
    let availableNodes = writable([]);
    let executionResults = writable(null);
    let isExecuting = writable(false);

    let svelteFlowInstance;
    let unsubscribeExecuting;
    //: The image the next run uses, chosen in the toolbar. Without one the workflow runs on
    //: whatever path its load_image node was saved with.
    let chosenImage = null;

    // Set context for child components - must be during initialization
    setContext('availableNodes', availableNodes);

    // Define custom node types for SvelteFlow
    const nodeTypes = {
        default: CustomNode
    };

    onMount(async () => {
        console.log('App onMount started');

        // Apply toolbar-hidden body class when default toolbar is suppressed
        if ($appConfig.hideToolbar) {
            document.body.classList.add('toolbar-hidden');
            document.documentElement.classList.remove('has-navbar-fixed-top');
        }

        try {
            availableNodes.set(await fetchCatalogue());
        } catch (error) {
            console.error('Could not load the node catalogue:', error);
        }

        // Dispatch a custom event whenever execution state changes so the
        // host app's custom header can react (e.g. disable the Run button)
        unsubscribeExecuting = isExecuting.subscribe(value => {
            window.dispatchEvent(new CustomEvent('agentui:statechange', {
                detail: { isExecuting: value }
            }));
        });
    });


    function initializeCanvas() {
        console.log('initializeCanvas called');
        nodes.set([createLoadImageNode()]);
        edges.set([]);

    }

    function handleNodeDrop(event) {
        const { nodeType, position } = event.detail;

        const category = getNodeCategory(nodeType);
        const newNode = {
            id: `${nodeType}-${Date.now()}`,
            type: 'default',
            position,
            data: {
                nodeType: nodeType,
                parameters: getDefaultParameters()
            },
            class: generateNodeClasses(nodeType, category),
            origin: [0.5, 0.0]
        };

        nodes.update(n => [...n, newNode]);

        // Check if there's a pending connection to auto-connect
        const pending = $pendingConnection;
        if (pending) {
            // Create auto-connection between pending handle and new node
            createAutoConnection(pending, newNode);
            clearPendingConnection();
        }
    }

    /** A new node sets nothing. Unset means the node's own default, on both sides of the wire. */
    function getDefaultParameters() {
        return {};
    }

    function getNodeCategory(nodeType) {
        const currentNodes = $availableNodes || {};
        return currentNodes[nodeType]?.category || 'Other';
    }

    function onNodeClick(event) {
        console.log('Node clicked:', event.detail.node.id);
        const node = event.detail.node;

        selectedNode.set(node);
        // Auto-open properties panel when a node is selected
        openSidebar('properties');
    }

    function onPaneClick() {
        // Close drawer when clicking on empty canvas
        closeSidebar();
        // Clear any pending connections
        clearPendingConnection();
    }

    function createAutoConnection(pending, newNode) {
        const { nodeId: pendingNodeId, handleId: pendingHandleId, handleType: pendingHandleType } = pending;

        let sourceNode, targetNode, sourceHandle, targetHandle;

        if (pendingHandleType === 'source') {
            // Pending node has output handle, new node should provide input
            sourceNode = pendingNodeId;
            targetNode = newNode.id;
            sourceHandle = pendingHandleId;
            targetHandle = getDefaultInput(newNode.data.nodeType);
        } else {
            // Pending node has input handle, new node should provide output
            sourceNode = newNode.id;
            targetNode = pendingNodeId;
            sourceHandle = getDefaultOutput(newNode.data.nodeType);
            targetHandle = pendingHandleId;
        }

        // Validate connection before creating
        if (sourceHandle && targetHandle) {
            const sourceNodeType = pendingHandleType === 'source' ?
                $nodes.find(n => n.id === pendingNodeId)?.data?.nodeType :
                newNode.data.nodeType;
            const targetNodeType = pendingHandleType === 'target' ?
                $nodes.find(n => n.id === pendingNodeId)?.data?.nodeType :
                newNode.data.nodeType;

            if (isValidConnection(sourceNodeType, sourceHandle, targetNodeType, targetHandle)) {
                const newEdge = {
                    id: `edge-${Date.now()}`,
                    source: sourceNode,
                    target: targetNode,
                    sourceHandle: sourceHandle,
                    targetHandle: targetHandle
                };

                edges.update(edges => [...edges, newEdge]);
                console.log('Auto-connection created:', newEdge);
            } else {
                console.warn('Auto-connection validation failed');
            }
        } else {
            console.warn('Could not determine handles for auto-connection');
        }
    }

    function onConnect(event) {
        const connection = event.detail.connection;

        // Get node types to determine correct handles
        const sourceNode = $nodes.find(n => n.id === connection.source);
        const targetNode = $nodes.find(n => n.id === connection.target);

        if (!sourceNode || !targetNode) {
            console.warn('Could not find source or target node');
            return;
        }

        const sourceNodeType = sourceNode.data?.nodeType;
        const targetNodeType = targetNode.data?.nodeType;

        // The handles the user actually dragged. Only fall back to the first port when a drag did
        // not name one -- picking the first regardless would make a multi-output node's second
        // port unreachable, and would send a handle nobody chose.
        const sourceHandle = connection.sourceHandle || getDefaultOutput(sourceNodeType);
        const targetHandle = connection.targetHandle || getDefaultInput(targetNodeType);

        // Validate connection types
        if (!isValidConnection(sourceNodeType, sourceHandle, targetNodeType, targetHandle)) {
            console.warn(`Invalid connection: ${sourceNodeType}.${sourceHandle} -> ${targetNodeType}.${targetHandle}`);
            alert(`Cannot connect ${sourceNodeType} output to ${targetNodeType} input: incompatible types`);
            return;
        }

        const newEdge = {
            id: `edge-${Date.now()}`,
            source: connection.source,
            target: connection.target,
            sourceHandle: sourceHandle,
            targetHandle: targetHandle
        };

        edges.update(edges => [...edges, newEdge]);
    }

    function isValidConnection(sourceNodeType, sourceHandle, targetNodeType, targetHandle) {
        const sourceNodeInfo = $availableNodes?.[sourceNodeType];
        const targetNodeInfo = $availableNodes?.[targetNodeType];

        if (!sourceNodeInfo || !targetNodeInfo) return false;
        if (!sourceHandle || !targetHandle) return false;

        // Ports are ordered lists of {name, type}, so they are found by name rather than indexed.
        // Indexing an array by a handle name yields undefined; indexing it by position yields a
        // port object, and comparing two of those compares references, which are never equal.
        const source = sourceNodeInfo.outputs?.find(port => port.name === sourceHandle);
        const target = targetNodeInfo.inputs?.find(port => port.name === targetHandle);

        if (!source || !target) return false;
        return source.type === target.type;
    }

    /** The name of a node's first output port, for a drag that did not name one. */
    function getDefaultOutput(nodeType) {
        return $availableNodes?.[nodeType]?.outputs?.[0]?.name || null;
    }

    /** The name of a node's first input port, for a drag that did not name one. */
    function getDefaultInput(nodeType) {
        return $availableNodes?.[nodeType]?.inputs?.[0]?.name || null;
    }

    function updateNodeParameters(nodeId, parameters) {
        nodes.update(n =>
            n.map(node =>
                node.id === nodeId
                    ? { ...node, data: { ...node.data, parameters } }
                    : node
            )
        );
        // And the selection, which the panel reads its current values from. Left stale, the panel
        // spreads the values it was opened with, so setting a second parameter discarded the
        // first.
        selectedNode.update(node =>
            node && node.id === nodeId
                ? { ...node, data: { ...node.data, parameters } }
                : node
        );
    }

    // Helper to strip execution state classes from a node's class string
    function stripExecutionClasses(classStr) {
        return classStr
            .replace(/\s*node-running/g, '')
            .replace(/\s*node-completed/g, '')
            .replace(/\s*node-error/g, '')
            .replace(/\s*node-has-output/g, '');
    }

    // Update a single node's class with execution state
    function updateNodeClass(nodeId, stateClass) {
        nodes.update(n => n.map(node => {
            if (node.id !== nodeId) return node;
            const baseClass = stripExecutionClasses(node.class);
            return { ...node, class: `${baseClass} ${stateClass}` };
        }));
    }

    async function executeWorkflow() {
        isExecuting.set(true);
        executionResults.set({ success: true, results: {} });

        // Reset all node classes before starting
        nodes.update(n => n.map(node => ({
            ...node,
            class: stripExecutionClasses(node.class)
        })));

        try {
            await stream(toDocument($nodes, $edges), chosenImage, (event) => {
                if (event.done) {
                    isExecuting.set(false);
                } else if (event.status === 'running') {
                    updateNodeClass(event.node, 'node-running');
                } else if (event.status === 'completed') {
                    updateNodeClass(event.node, 'node-completed node-has-output');
                    executionResults.update(r => ({
                        ...r,
                        results: { ...r.results, [event.node]: event.output }
                    }));
                } else if (event.status === 'failed') {
                    if (event.node) updateNodeClass(event.node, 'node-error');
                    executionResults.update(r => ({ ...r, success: false, error: event.error }));
                    isExecuting.set(false);
                }
            });
        } catch (error) {
            executionResults.update(r => ({ ...r, success: false, error: error.message }));
        } finally {
            isExecuting.set(false);
        }
    }

    /** Pick the image the next run uses, instead of whatever path the workflow was saved with. */
    function chooseImage(event) {
        chosenImage = event.target.files[0] || null;
    }

    function exportWorkflow() {
        const dataStr = JSON.stringify(toDocument($nodes, $edges), null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        const url = URL.createObjectURL(dataBlob);

        const link = document.createElement('a');
        link.href = url;
        link.download = 'workflow.json';
        link.click();

        URL.revokeObjectURL(url);
    }

    function importWorkflow(event) {
        const file = event.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (e) => {
            try {
                const canvas = fromDocument(JSON.parse(e.target.result), $availableNodes);
                nodes.set(canvas.nodes.map(node => ({
                    ...node, type: 'default',
                    class: generateNodeClasses(
                        node.data.nodeType, $availableNodes[node.data.nodeType]?.category || '')
                })));
                edges.set(canvas.edges);
            } catch (error) {
                alert('Invalid workflow file');
            }
        };
        reader.readAsText(file);
    }

    function clearWorkflow() {
        initializeCanvas();
        selectedNode.set(null);
        executionResults.set(null);
    }

    onDestroy(() => {
        if (unsubscribeExecuting) unsubscribeExecuting();
    });

</script>

<SvelteFlowProvider>
            <Toolbar
                    {executeWorkflow}
                    {exportWorkflow}
                    {importWorkflow}
                    isExecuting={$isExecuting}
                    {chooseImage}
                    {chosenImage}
            />


<!-- Left-side node palette (always visible) -->
<NodePalettePanel {availableNodes} />

<!-- Main app layout without fixed sidebar -->
<div class="main-content">
    <div id="studio">
        <FlowDropZone on:nodedrop={handleNodeDrop}>
            <SvelteFlow
                    {nodes}
                    {edges}
                    {nodeTypes}
                    proOptions={{ hideAttribution: true }}
                    bind:this={svelteFlowInstance}
                    on:nodeclick={onNodeClick}
                    on:paneclick={onPaneClick}
                    on:connect={onConnect}
                    maxZoom={1}
                    fitView
                    fitViewOptions={{
                      maxZoom: 1,      // Prevents zooming in beyond 100%
                      padding: 0.2     // Adds some padding around nodes
                    }}
            >
<!--                <Controls />-->
                <Background variant="dots" />
            </SvelteFlow>
        </FlowDropZone>
    </div>
</div>

<!-- Drawer sidebar for context-aware panels -->
<DrawerSidebar
    {availableNodes}
    selectedNode={$selectedNode}
    {updateNodeParameters}
    executionResults={$executionResults}
/>


</SvelteFlowProvider>