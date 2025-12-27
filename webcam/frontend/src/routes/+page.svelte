<script lang="ts">
  import { onMount, onDestroy } from 'svelte';
  import type { Fields, PipelineInfo } from '$lib/types';
  import { PipelineMode } from '$lib/types';
  import ImagePlayer from '$lib/components/ImagePlayer.svelte';
  import VideoInput from '$lib/components/VideoInput.svelte';
  import ImageInput from '$lib/components/ImageInput.svelte';
  import Button from '$lib/components/Button.svelte';
  import PipelineOptions from '$lib/components/PipelineOptions.svelte';
  import Spinner from '$lib/icons/spinner.svelte';
  import Warning from '$lib/components/Warning.svelte';
  import Logging from '$lib/components/Logging.svelte';
  import { lcmLiveStatus, lcmLiveActions, LCMLiveStatus } from '$lib/lcmLive';
  import { mediaStreamActions, onFrameChangeStore, referenceImageStore, referenceImageSent } from '$lib/mediaStream';
  import { getPipelineValues, deboucedPipelineValues } from '$lib/store';
  
  import { FramePusher } from '$lib/framePusher';

  let pipelineParams: Fields;
  let pipelineInfo: PipelineInfo;
  let pageContent: string;
  let isImageMode: boolean = false;
  let maxQueueSize: number = 0;
  let currentQueueSize: number = 0;
  let queueCheckerRunning: boolean = false;
  let warningMessage: string = '';
  let loggingMessage: string = '';

  let fps: number = 25;

  const framePusher = new FramePusher(lcmLiveActions.send);

  $: framePusher.setFPS(fps);

  onMount(() => {
    getSettings();
  });

  onDestroy(() => {
    framePusher.stop();
  });

  async function getSettings() {
    const settings = await fetch(`/api/settings`).then((r) => r.json());
    pipelineParams = settings.input_params.properties;
    pipelineInfo = settings.info.properties;
    isImageMode = pipelineInfo.input_mode.default === PipelineMode.IMAGE;
    maxQueueSize = settings.max_queue_size;
    pageContent = settings.page_content;
    console.log(pipelineParams);
    toggleQueueChecker(true);
  }
  function toggleQueueChecker(start: boolean) {
    queueCheckerRunning = start && maxQueueSize > 0;
    if (start) {
      getQueueSize();
    }
  }
  async function getQueueSize() {
    if (!queueCheckerRunning) {
      return;
    }
    const data = await fetch(`/api/queue`).then((r) => r.json());
    currentQueueSize = data.queue_size;
    setTimeout(getQueueSize, 10000);
  }

  function getSreamdata() {
    if (isImageMode) {
      return [$onFrameChangeStore?.blob];
    } else {
      return [];
    }
  }

  async function sendReference() {
    const refBlob = $referenceImageStore?.blob;
    if (!refBlob || refBlob.size === 0) {
      warningMessage = 'Please upload reference portrait first.';
      return;
    }

    const form = new FormData();
    form.append("ref_image", refBlob, "reference.jpg");

    const res = await fetch(`/api/upload_reference_image`, {
      method: "POST",
      body: form
    });

    if (!res.ok) {
      warningMessage = "Failed to upload reference image.";
      return;
    }
    referenceImageSent.set(true);
    loggingMessage = 'Successfully uploaded reference portrait.';
    warningMessage = "";
  }

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED && $lcmLiveStatus !== LCMLiveStatus.PAUSED;
  $: if ($lcmLiveStatus === LCMLiveStatus.TIMEOUT) {
    warningMessage = 'Session timed out. Please try again.';
    framePusher.stop();
  }

  let disabled = false;
  async function toggleLcmLive() {
    try {
      const refFlag = $referenceImageSent;
      if (!refFlag) {
        warningMessage = 'Please upload reference portrait first.';
        return;
      }

      if (!isLCMRunning) {
        if (isImageMode) {
          await mediaStreamActions.enumerateDevices();
          await mediaStreamActions.start();
        }
        disabled = true;
        
        await lcmLiveActions.start(); 
        
        framePusher.start();

        disabled = false;
        toggleQueueChecker(false);
      } else {
        if (isImageMode) {
          mediaStreamActions.stop();
        }
        disabled = true;
        
        framePusher.stop();
        lcmLiveActions.pause();
        
        disabled = false;
        toggleQueueChecker(true);
      }
    } catch (e) {
      warningMessage = e instanceof Error ? e.message : '';
      disabled = false;
      framePusher.stop();
      toggleQueueChecker(true);
    }
  }
</script>

<svelte:head>
  <script
    src="https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.9/iframeResizer.contentWindow.min.js"
  ></script>
</svelte:head>

<main class="container mx-auto flex max-w-5xl flex-col gap-3 px-4 py-4">
  <Warning bind:message={warningMessage}></Warning>
  <Logging bind:message={loggingMessage}></Logging>
  <article class="text-center">
    {#if pageContent}
      {@html pageContent}
    {/if}
    {#if maxQueueSize > 0}
      <p class="text-sm">
        There are <span id="queue_size" class="font-bold">{currentQueueSize}</span>
        user(s) sharing the same GPU, affecting real-time performance. Maximum queue size is {maxQueueSize}.
        <a
          href="https://huggingface.co/spaces/radames/Real-Time-Latent-Consistency-Model?duplicate=true"
          target="_blank"
          class="text-blue-500 underline hover:no-underline">Duplicate</a
        > and run it on your own GPU.
      </p>
    {/if}
  </article>
  {#if pipelineParams}
    <article class="my-3 grid grid-cols-1 sm:grid-cols-[340px_1fr] gap-6">

      <div class="flex flex-col gap-6">
        <div class="aspect-square w-full">
          <ImageInput 
            width={Number(pipelineParams.width.default)}
            height={Number(pipelineParams.height.default)}
          />
        </div>
        {#if isImageMode}
          <div class="aspect-square w-full">
            <VideoInput
              width={Number(pipelineParams.width.default)}
              height={Number(pipelineParams.height.default)}
            />
          </div>
        {/if}
      </div>

      <div class="flex flex-col gap-2">
        <div class="w-full aspect-square">
          <ImagePlayer />
        </div>
        <div class="sm:col-span-2 border border-slate-300 rounded-lg p-4 flex gap-4 justify-center">
          <div class="flex items-center gap-4 flex-1">
              <span class="text-sm font-bold whitespace-nowrap">Driving FPS: {fps}</span>

              <input 
                  type="range" 
                  min="1" 
                  max="30" 
                  step="1" 
                  bind:value={fps} 
                  class="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer dark:bg-gray-700"
              />
          </div>
          <Button on:click={sendReference} disabled={$referenceImageSent} classList="px-2 py-2 text-sm">Fuse reference</Button>

          <Button on:click={toggleLcmLive} {disabled} classList="px-2 py-2 text-sm">
            {#if isLCMRunning}
              Stop Animation
            {:else}
              Start Animation
            {/if}
          </Button>

        </div>
      </div>
    </article>
  {:else}
    <!-- loading -->
    <div class="flex items-center justify-center gap-3 py-48 text-2xl">
      <Spinner classList={'animate-spin opacity-50'}></Spinner>
      <p>Loading...</p>
    </div>
  {/if}
</main>

<style lang="postcss">
  :global(html) {
    @apply text-black dark:bg-gray-900 dark:text-white;
  }
</style>
