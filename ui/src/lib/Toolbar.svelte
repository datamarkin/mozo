<script>
  import { appConfig } from './stores.js';

  export let executeWorkflow;
  export let exportWorkflow;
  export let importWorkflow;
  export let isExecuting;

  let fileInput;

  function handleImportClick() {
    fileInput.click();
  }
</script>

{#if !$appConfig.hideToolbar}
<nav class="navbar is-fixed-top border-bottom" aria-label="main navigation">
  <div class="navbar-brand">
    <div class="navbar-item">
        </div>
    <div class="navbar-item">

    </div>
  </div>

  <div class="navbar-menu">
    <div class="navbar-end">
      <div class="navbar-item">
        <div class="buttons">
          <button class="button is-dark" on:click={executeWorkflow} disabled={isExecuting}>
            {isExecuting ? 'Running...' : 'Run workflow'}
          </button>

          <button class="button is-dark" on:click={exportWorkflow}>Export</button>
          <button class="button" on:click={handleImportClick}>Import</button>
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
{/if}
