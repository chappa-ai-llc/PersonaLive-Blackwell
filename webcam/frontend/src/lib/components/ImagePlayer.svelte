<script lang="ts">
  import { onMount } from 'svelte';
  import { lcmLiveStatus, LCMLiveStatus, latestFrameUrl } from '$lib/lcmLive';
  import { getPipelineValues } from '$lib/store';
  import Button from '$lib/components/Button.svelte';
  import Floppy from '$lib/icons/floppy.svelte';
  import { snapImage } from '$lib/utils';

  $: isLCMRunning = $lcmLiveStatus !== LCMLiveStatus.DISCONNECTED;

  let canvasEl: HTMLCanvasElement;
  let ctx: CanvasRenderingContext2D | null = null;

  onMount(() => {
    if (canvasEl) {
      ctx = canvasEl.getContext('2d', { alpha: false });
    }
  });

  $: if ($latestFrameUrl && ctx && canvasEl) {
    drawFrame($latestFrameUrl);
  }

  async function drawFrame(url: string) {
    if (!ctx || !canvasEl) return;

    const img = new Image();
    img.src = url;
    
    try {
      await img.decode();
      
      // 动态调整 Canvas 尺寸
      if (canvasEl.width !== img.naturalWidth || canvasEl.height !== img.naturalHeight) {
          canvasEl.width = img.naturalWidth;
          canvasEl.height = img.naturalHeight;
      }
      
      ctx.drawImage(img, 0, 0);
      
    } catch (e) {
      // 忽略解码错误
    }
  }

  async function takeSnapshot() {
    if (isLCMRunning && canvasEl) {
      canvasEl.toBlob(async (blob) => {
        if (!blob) return;
        const tempImg = new Image();
        tempImg.src = URL.createObjectURL(blob);
        await tempImg.decode(); 
        await snapImage(tempImg, {
            prompt: getPipelineValues()?.prompt,
            negative_prompt: getPipelineValues()?.negative_prompt,
            seed: getPipelineValues()?.seed,
            guidance_scale: getPipelineValues()?.guidance_scale
        });
      }, 'image/jpeg');
    }
  }
</script>

<div class="relative mx-auto aspect-square self-center overflow-hidden rounded-lg border border-slate-300 dark:border-slate-700 bg-slate-50 dark:bg-black transition-colors duration-200">
  <div class="absolute top-2 left-2 z-30 bg-black/50 text-white text-xs font-medium px-2 py-0.5 rounded-md backdrop-blur-sm shadow">
    🎞 Animation
  </div>

  <canvas
    bind:this={canvasEl}
    width="512"
    height="512"
    class="w-full h-full object-contain block"
  ></canvas>

  {#if !$latestFrameUrl}
      <div class="absolute inset-0 flex flex-col items-center justify-center bg-slate-100 dark:bg-slate-900 text-slate-400 dark:text-slate-500 z-10 transition-colors duration-200">
         {#if isLCMRunning}
            <span>Wait...</span>
         {:else}
             <img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=" alt="" class="w-full h-full opacity-0"/>
         {/if}
      </div>
  {/if}

  <div class="absolute bottom-1 right-1 z-20">
    <Button
      on:click={takeSnapshot}
      disabled={!isLCMRunning}
      title={'Take Snapshot'}
      classList={'text-sm ml-auto text-white p-1 shadow-lg rounded-lg opacity-50'}
    >
      <Floppy classList={''} />
    </Button>
  </div>
</div>