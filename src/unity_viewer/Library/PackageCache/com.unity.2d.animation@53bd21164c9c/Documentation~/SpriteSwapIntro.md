# Introduction to Sprite Swap
This page introduces what's Sprite Swap, its different uses and its limitations. **Sprite Swap** refers to changing the rendered Sprite of a GameObject at runtime, which is useful when animating the Sprites that make up a 2D actor or other GameObjects.

For example, you can [swap the individual Sprites](ex-sprite-swap.md#part-swap) that make up an animated actor to create multiple actors that [share the same skeleton](ex-skeleton-sharing.md) (requires the [PSD Importer package](https://docs.unity3d.com/Packages/com.unity.2d.psdimporter@latest); or [create animation clips](ex-sprite-swap.md#animated-swap) by swapping the Sprites at runtime.

You can import [sample projects](Examples.md) for the 2D Animation package by selecting the option in the 2D Animation package window. Refer to the individual [Sprite Swap examples](ex-sprite-swap.md) pages for more information about these samples.

## Required assets and components

Sprite Swap requires the following Assets and component, which are available with the 2D Animation package:

* [Sprite Library Asset](SL-Asset.md):  The Sprite Library Asset contains a set of selected Sprites which are assigned to different [Categories](SL-Editor.md#categories) and [Labels](SL-Editor.md#labels).
  <br/>

* [Sprite Library component](SL-component.md): The Sprite Library component determines which Sprite Library Asset a GameObject refers to.
  <br/>

* [Sprite Resolver component](SL-Resolver.md): The Sprite Resolver component requests a Sprite registered to the Sprite Library Asset by referring to the **Category** and **Label** value of the desired Sprite.

## Technical limitations

The following are technical limitations which you should keep in mind when using Sprite Swap.

### Skeletal animation limitations

If you want to [animate your actor](Animating-actor.md) and use Sprite Swap with skeletal animation, both sprites that are swapped must have an identical skeleton. Use the [Copy and Paste tools of the Skinning Editor](SkinEdToolsShortcuts.md#copy-and-paste-behavior) to duplicate the bone and skeleton data from one sprite to another to ensure they will swap correctly.

### Animator limitations
In a single [Animator Controller](https://docs.unity3d.com/Manual/AnimatorControllers.html), you can't have one [Animation Clip](https://docs.unity3d.com/Manual/AnimationClips.html) animating the [Sprite Renderer’s](https://docs.unity3d.com/Manual/class-SpriteRenderer.html) assigned sprite while another [Animation Clip](https://docs.unity3d.com/Manual/AnimationClips.html) animates the [Sprite Resolver’s](SL-Resolver.md) sprite hash. If these two clips are in the same [Animator Controller](https://docs.unity3d.com/Manual/AnimatorControllers.html), they will conflict with each other and cause unwanted playback results.

Use the following recommended methods to resolve this issue.

1. The first method is to separate the [Animation Clips](https://docs.unity3d.com/Manual/AnimationClips.html) into separate [Animator Controllers](https://docs.unity3d.com/Manual/AnimatorControllers.html) that contain only clips that animate either a [Sprite Renderer’s](https://docs.unity3d.com/Manual/class-SpriteRenderer.html) sprite or the [Sprite Resolver’s](SL-Resolver.md) sprite hash but not both types in the same [Animator Controller](https://docs.unity3d.com/Manual/AnimatorControllers.html).
   <br/>

2. The second method is to update all [Animation Clips](https://docs.unity3d.com/Manual/AnimationClips.html) to the same type so that they can all remain in a single [Animator Controller](https://docs.unity3d.com/Manual/AnimatorControllers.html). To do so, convert all clips animating a [Sprite Renderer’s](https://docs.unity3d.com/Manual/class-SpriteRenderer.html) sprite to animating a [Sprite Resolver’s](SL-Resolver.md) sprite hash, or vice versa.

## Additional resources
- [Animation](https://docs.unity3d.com/Manual/AnimationSection.html)
- [Skinning Editor](SkinningEditor.md)
- [PSD Importer package](https://docs.unity3d.com/Packages/com.unity.2d.psdimporter@latest)
