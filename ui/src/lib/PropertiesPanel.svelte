<script>
    /**
     * The form for one node's parameters.
     *
     * Every widget comes from the `kind` the server declared -- `int`, `float`, `str`, `bool`,
     * `color`, or `select` with its options. Nothing here guesses from the value, which is the
     * point of the catalogue being derived from the node's own signature: a parameter that has
     * never been set still knows what it is.
     *
     * A parameter marked `optional` may be left blank, and blank is sent as nothing rather than as
     * a zero or an empty string. That is what lets a thickness scale itself to the image and a
     * crop run to the image's own edge.
     *
     * `source` is the one kind that is not just a box: the value is a path, and a person at a
     * browser cannot write the server's path to a file sitting on their own machine. So it offers
     * a file to upload as well, and says which of the two the next run will use.
     */
    import { formatParameterLabel } from './utils.js';
    import { chosenImage } from './stores.js';

    export let selectedNode;
    export let updateNodeParameters;
    export let availableNodes;
    export let executionResults;
    export let hideTitle = false;

    $: nodeInfo = availableNodes?.[selectedNode?.data?.nodeType] || {};
    $: fields = nodeInfo.parameters || [];
    $: set = selectedNode?.data?.parameters || {};

    /** What a field is showing: what the workflow saved, or the node's own default. */
    function valueOf(field) {
        return field.name in set ? set[field.name] : field.default;
    }

    function change(field, value) {
        updateNodeParameters(selectedNode.id, { ...set, [field.name]: value });
    }

    /** An empty box on an optional field means "unset", which is a value the server understands. */
    function changeNumber(field, raw, whole) {
        if (raw === '' && field.optional) return change(field, null);
        const parsed = whole ? parseInt(raw, 10) : parseFloat(raw);
        change(field, Number.isNaN(parsed) ? field.default : parsed);
    }

    function changeText(field, raw) {
        change(field, raw === '' && field.optional ? null : raw);
    }

    /** Take the first image from a picker or a drop, whichever the file arrived by. */
    function chooseFile(files) {
        const file = [...(files || [])].find(candidate => candidate.type.startsWith('image/'));
        if (file) chosenImage.set(file);
    }

    let dragging = false;

    $: result = executionResults?.results?.[selectedNode?.id];
</script>

{#if !hideTitle}
    <div class="block mb-1"><strong>Node properties</strong></div>
{/if}

<nav class="panel">
    {#if selectedNode}
        <div class="panel-block">
            <div>
                <p class="is-size-8 has-text-weight-bold">{selectedNode.data.nodeType}</p>
                <p class="is-size-7 has-text-grey">{nodeInfo.description || ''}</p>
            </div>
        </div>

        {#each fields as field (field.name)}
            {@const value = valueOf(field)}
            <div class="panel-block">
                <div class="field" style="width: 100%;">
                    <label class="label is-small" for={field.name}>
                        {formatParameterLabel(field.name)}
                        {#if field.optional}<span class="has-text-grey is-size-7"> — optional</span>{/if}
                    </label>
                    <div class="control">
                        {#if field.kind === 'bool'}
                            <label class="checkbox">
                                <input id={field.name} type="checkbox" checked={!!value}
                                       on:change={(e) => change(field, e.target.checked)} />
                                {formatParameterLabel(field.name)}
                            </label>
                        {:else if field.kind === 'select'}
                            <div class="select is-small is-fullwidth">
                                <select id={field.name} value={value}
                                        on:change={(e) => change(field, e.target.value)}>
                                    {#each field.options as option}
                                        <option value={option}>{option}</option>
                                    {/each}
                                </select>
                            </div>
                        {:else if field.kind === 'color'}
                            <div class="is-flex is-align-items-center">
                                <input id={field.name} type="color" value={value || '#00ff00'}
                                       on:input={(e) => change(field, e.target.value)}
                                       style="width: 3rem; height: 2rem; padding: 0; border: none;" />
                                {#if field.optional}
                                    <button class="button is-small is-ghost ml-2"
                                            on:click={() => change(field, null)}>
                                        by class
                                    </button>
                                {/if}
                            </div>
                        {:else if field.kind === 'source'}
                            <div class="source" class:dragging
                                 role="button" tabindex="-1"
                                 on:dragover|preventDefault={() => (dragging = true)}
                                 on:dragleave={() => (dragging = false)}
                                 on:drop|preventDefault={(e) => {
                                     dragging = false;
                                     chooseFile(e.dataTransfer.files);
                                 }}>
                                <label class="button is-small is-fullwidth">
                                    {$chosenImage ? $chosenImage.name : 'Drop or choose an image'}
                                    <input type="file" accept="image/*" style="display: none;"
                                           on:change={(e) => chooseFile(e.target.files)} />
                                </label>
                            </div>
                            {#if $chosenImage}
                                <p class="is-size-7 has-text-grey mt-1">
                                    The next run reads this file.
                                    <button class="is-link-ish" on:click={() => chosenImage.set(null)}>
                                        Use the path instead
                                    </button>
                                </p>
                            {/if}
                            <input id={field.name} class="input is-small mt-1" type="text"
                                   value={value === null || value === undefined ? '' : value}
                                   placeholder="or a path on the server"
                                   disabled={!!$chosenImage}
                                   on:input={(e) => changeText(field, e.target.value)} />
                        {:else if field.kind === 'int' || field.kind === 'float'}
                            <input id={field.name} class="input is-small" type="number"
                                   step={field.kind === 'int' ? '1' : '0.05'}
                                   value={value === null || value === undefined ? '' : value}
                                   placeholder={field.optional ? 'automatic' : ''}
                                   on:input={(e) => changeNumber(field, e.target.value,
                                                                 field.kind === 'int')} />
                        {:else}
                            <input id={field.name} class="input is-small" type="text"
                                   value={value === null || value === undefined ? '' : value}
                                   placeholder={field.optional ? 'unset' : ''}
                                   on:input={(e) => changeText(field, e.target.value)} />
                        {/if}
                    </div>
                </div>
            </div>
        {/each}

        {#if result}
            <div class="panel-block">
                <div style="width: 100%;">
                    <p class="is-size-7 has-text-weight-bold mb-1">Result</p>
                    {#if typeof result === 'string' && result.startsWith('data:image')}
                        <img src={result} alt="what this node produced" style="max-width: 100%;" />
                    {:else if result && result.depth}
                        <img src={result.depth} alt="depth map" style="max-width: 100%;" />
                        <p class="is-size-7 has-text-grey">
                            {result.min.toFixed(2)} to {result.max.toFixed(2)}
                        </p>
                    {:else if Array.isArray(result)}
                        <p class="is-size-7 has-text-grey">{result.length} results</p>
                        <pre class="is-size-7" style="max-height: 12rem; overflow: auto;">{JSON.stringify(result.slice(0, 10), null, 1)}</pre>
                    {:else}
                        <pre class="is-size-7" style="max-height: 12rem; overflow: auto;">{JSON.stringify(result, null, 1)}</pre>
                    {/if}
                </div>
            </div>
        {/if}
    {:else}
        <div class="panel-block">
            <p class="is-size-7 has-text-grey">Select a node to edit it.</p>
        </div>
    {/if}
</nav>

<style>
    /* The drop target. Dashed while idle so it reads as "put something here" rather than as a
       button that has lost its border, and filled while a file is over it. */
    .source {
        border: 1px dashed #c9c9c9;
        border-radius: 4px;
        padding: 0.4rem;
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }

    .source.dragging {
        border-color: #485fc7;
        background-color: #f0f3ff;
    }

    /* A button that reads as a link, for an action that undoes rather than commits. */
    .is-link-ish {
        background: none;
        border: none;
        padding: 0;
        color: #485fc7;
        cursor: pointer;
        font: inherit;
        text-decoration: underline;
    }
</style>
