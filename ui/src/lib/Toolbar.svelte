<script>
  /**
   * What you do to a workflow as a whole: run it, save it, open one.
   *
   * The image a run reads is not here. It belongs to the node that reads it, where the person is
   * already looking when they wonder what it will run on -- see the `source` widget in
   * PropertiesPanel.
   */
  export let executeWorkflow;
  export let exportWorkflow;
  export let importWorkflow;
  export let isExecuting;

  let fileInput;
</script>

<nav class="navbar is-fixed-top border-bottom" aria-label="main navigation">
  <div class="navbar-menu">
    <div class="navbar-end">
      <div class="navbar-item">
        <div class="buttons">
          <button class="button is-dark" on:click={executeWorkflow} disabled={isExecuting}>
            {isExecuting ? 'Running...' : 'Run workflow'}
          </button>

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
