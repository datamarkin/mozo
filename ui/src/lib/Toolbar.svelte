<script>
  /**
   * What you do to a workflow as a whole: test it, run it, save it, open one.
   *
   * **Two verbs, because they are two different things.** Test puts one item through the graph and
   * draws every node's output, which is what you want while wiring. Run puts the whole source
   * through and draws none of them -- see `process` in api.js for why it cannot.
   *
   * The file a run reads is not here. It belongs to the node that reads it, where the person is
   * already looking when they wonder what it will run on -- see the `source` widget in
   * PropertiesPanel.
   */
  export let testWorkflow;
  export let runWorkflow;
  export let cancelRun;
  export let exportWorkflow;
  export let importWorkflow;
  export let isExecuting;
  export let isRunning;

  let fileInput;
</script>

<nav class="navbar is-fixed-top border-bottom" aria-label="main navigation">
  <div class="navbar-menu">
    <div class="navbar-end">
      <div class="navbar-item">
        <div class="buttons">
          <button class="button" on:click={testWorkflow}
                  disabled={isExecuting || isRunning}
                  title="One item through the graph, with every node's output shown">
            {isExecuting ? 'Testing...' : 'Test'}
          </button>

          {#if isRunning}
            <button class="button is-danger is-light" on:click={cancelRun}>Cancel</button>
          {:else}
            <button class="button is-dark" on:click={runWorkflow} disabled={isExecuting}
                    title="Every item the input produces">
              Run
            </button>
          {/if}

          <button class="button is-dark" on:click={exportWorkflow}>Export</button>
          <button class="button" on:click={() => fileInput.click()}>Import</button>
        </div>
      </div>
    </div>
  </div>

  <input
    bind:this={fileInput}
    type="file"
    accept=".json"
    class="is-hidden"
    on:change={importWorkflow}
  />
</nav>
