<script>
    /**
     * What a run in progress looks like: how far it has got, and a glance at one node.
     *
     * Not the canvas. Test draws every node's output at full size on the graph itself; a run has
     * ten thousand of those and no room for any of them, so this is one counter and one small
     * picture in a corner. The picture is a JPEG a fraction the size of the ones the canvas gets,
     * arriving at most five times a second however fast the run goes.
     */
    import { running } from './stores.js';

    $: rate = $running?.seconds ? Math.round($running.items / $running.seconds) : null;
</script>

{#if $running}
    <aside class="run" aria-live="polite">
        {#if $running.preview}
            <img src={$running.preview} alt="the run in progress" />
        {/if}
        <div class="tally">
            <strong>{$running.items.toLocaleString()}</strong>
            <span>{$running.done ? 'items' : 'items so far'}</span>
            {#if rate}<span class="rate">{rate.toLocaleString()}/s</span>{/if}
            {#if $running.node}<span class="node">showing {$running.node}</span>{/if}
        </div>
    </aside>
{/if}

<style>
    .run {
        position: fixed;
        bottom: 1rem;
        left: 1rem;
        z-index: 30;
        background: rgba(20, 20, 22, 0.92);
        color: #f4f4f5;
        border-radius: 6px;
        padding: 0.5rem;
        display: flex;
        gap: 0.75rem;
        align-items: center;
        max-width: min(28rem, calc(100vw - 2rem));
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.35);
    }

    .run img {
        display: block;
        width: 10rem;
        max-width: 40vw;
        border-radius: 3px;
    }

    .tally {
        display: flex;
        flex-direction: column;
        gap: 0.1rem;
        font-size: 0.8rem;
        /* Digits that change every frame must not shuffle the layout under them. */
        font-variant-numeric: tabular-nums;
    }

    .tally strong { font-size: 1.15rem; }
    .rate, .node { opacity: 0.65; }
    .node { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
</style>
