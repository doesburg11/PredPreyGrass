using System.Collections.Generic;
using UnityEditor.Animations;
using UnityEditor.AssetImporters;
using UnityEngine;

namespace UnityEditor.U2D.Aseprite
{
    internal static class AnimatorControllerGeneration
    {
        public static void Generate(AssetImportContext ctx, string assetName, GameObject rootGameObject, bool generateModelPrefab)
        {
            var assetObjects = new List<Object>();
            ctx.GetObjects(assetObjects);

            var animationClips = new List<AnimationClip>();
            foreach (var obj in assetObjects)
            {
                if (obj is AnimationClip clip)
                    animationClips.Add(clip);
            }

            if (animationClips.Count == 0)
                return;

            var controller = new AnimatorController();
            controller.name = assetName;
            controller.AddLayer("Base Layer");

            foreach (var clip in animationClips)
                controller.AddMotion(clip);

            ctx.AddObjectToAsset(controller.name + "_Controller", controller);
            foreach (var layer in controller.layers)
            {
                var stateMachine = layer.stateMachine;
                ctx.AddObjectToAsset(stateMachine.name + "_StateMachine", stateMachine);

                foreach (var state in stateMachine.states)
                    ctx.AddObjectToAsset(state.state.name + "_State", state.state);
            }

            if (generateModelPrefab)
            {
                var animator = rootGameObject.AddComponent<Animator>();
                AnimatorController.SetAnimatorController(animator, controller);
            }
        }
    }
}
