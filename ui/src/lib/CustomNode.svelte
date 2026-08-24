<script>
  import { Handle, Position, useEdges } from '@xyflow/svelte';
  import { getContext } from 'svelte';
  import CategoryIcon from './CategoryIcon.svelte';
  import { pendingConnection } from './stores.js';

  export let data;
  export let id;

  // Get edges from SvelteFlow context
  const edges = useEdges();

  // Get available nodes context to determine required/optional inputs
  const availableNodes = getContext('availableNodes') || {};

  $: nodeInfo = $availableNodes[data.nodeType] || {};
  // Ordered lists, exactly as the server derived them from the node's signature. Every input is
  // required: a node whose input carried a default would be refused at registration, because an
  // input arrives over a connection and a default is a value nothing can produce.
  $: inputPorts = nodeInfo.inputs || [];
  $: outputPorts = nodeInfo.outputs || [];

  // Check connection state for each handle
  $: connectedInputs = $edges.filter(edge => edge.target === id).map(edge => edge.targetHandle);
  $: connectedOutputs = $edges.filter(edge => edge.source === id).map(edge => edge.sourceHandle);

  // Check if all inputs/outputs are connected
  $: allInputsConnected = inputPorts.length > 0 &&
                          inputPorts.every(port => connectedInputs.includes(port.name));
  $: allOutputsConnected = outputPorts.length > 0 &&
                           outputPorts.every(port => connectedOutputs.includes(port.name));

  // Reactive statement for node connection classes - Svelte will track this
  $: nodeConnectionClasses = [
    allInputsConnected && 'inputs-connected',
    allOutputsConnected && 'outputs-connected'
  ].filter(Boolean).join(' ');

  // Every input is required: an input carrying a default is refused when the node registers,
  // because an input arrives over a connection and a default is a value nothing can produce.


  function isInputConnected(inputName) {
    return connectedInputs.includes(inputName);
  }

  function isOutputConnected(outputName) {
    return connectedOutputs.includes(outputName);
  }

  function getInputHandleClass(inputName) {
    // Every input is required: an input carrying a default is refused when the node registers,
    // because an input arrives over a connection and a default is a value nothing can produce.
    let baseClass = 'handle-required';

    // Add pending class if this handle is pending connection
    const pending = $pendingConnection;
    if (pending && pending.nodeId === id && pending.handleId === inputName && pending.handleType === 'target') {
      baseClass += ' handle-pending';
    }

    return baseClass;
  }

  function getOutputHandleClass(outputName) {
    let baseClass = 'handle-output';

    // Add pending class if this handle is pending connection
    const pending = $pendingConnection;
    if (pending && pending.nodeId === id && pending.handleId === outputName && pending.handleType === 'source') {
      baseClass += ' handle-pending';
    }

    return baseClass;
  }

</script>



<!-- Input handles at top -->
{#each inputPorts as inputPort, index}
  {@const inputName = inputPort.name}
  {@const leftPosition = inputPorts.length === 1 ? 50 : (100 / (inputPorts.length + 1)) * (index + 1)}
  {@const isConnected = connectedInputs.includes(inputName)}
  {@const portType = inputPort.type}

  <!-- Handle with automatic color styling based on port type (port-type-{type} class) -->
  <!-- See custom.css for port type color definitions -->
  <Handle
    type="target"
    position={Position.Top}
    id={inputName}
    class="input-handle port-type-{portType} {getInputHandleClass(inputName)} {isConnected ? 'handle-connected' : ''}"
    style="position: absolute; left: {leftPosition}%; transform: translateX(-50%);"
  />
{/each}

<div class="panel-block has-text-left {nodeConnectionClasses}" draggable="true">
  <span class="panel-icon">
    <CategoryIcon category={nodeInfo.category || 'Other'} size={16} />
  </span>
  <div>
    <p class="is-size-8 has-text-weight-bold">{data.nodeType}</p>
    <p class="is-size-7 has-text-grey">
      {#if data.parameters && Object.keys(data.parameters).length > 0}
        {Object.keys(data.parameters).length} parameters
      {:else}
        {nodeInfo.description || 'Process image data'}
      {/if}
    </p>
  </div>
</div>

<!-- Output handles at bottom -->
{#each outputPorts as outputPort, index}
  {@const outputName = outputPort.name}
  {@const leftPosition = outputPorts.length === 1 ? 50 : (100 / (outputPorts.length + 1)) * (index + 1)}
  {@const isConnected = connectedOutputs.includes(outputName)}
  {@const portType = outputPort.type}

  <!-- Handle with automatic color styling based on port type (port-type-{type} class) -->
  <!-- See custom.css for port type color definitions -->
  <Handle
    type="source"
    position={Position.Bottom}
    id={outputName}
    class="output-handle port-type-{portType} {getOutputHandleClass(outputName)} {isConnected ? 'handle-connected' : ''}"
    style="position: absolute; left: {leftPosition}%; transform: translateX(-50%);"
  />
{/each}

