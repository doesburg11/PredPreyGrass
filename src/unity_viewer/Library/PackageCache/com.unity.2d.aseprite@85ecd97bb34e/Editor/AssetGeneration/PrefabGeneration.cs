using System;
using System.Collections.Generic;
using UnityEditor.AssetImporters;
using UnityEngine;
using UnityEngine.Rendering;

namespace UnityEditor.U2D.Aseprite
{
    internal static class PrefabGeneration
    {
        public static void Generate(
            AssetImportContext ctx,
            TextureGenerationOutput output,
            List<Layer> layers,
            Dictionary<int, GameObject> layerIdToGameObject,
            Vector2Int canvasSize,
            AsepriteImporterSettings importSettings,
            ref UnityEngine.Object mainAsset,
            out GameObject rootGameObject)
        {
            rootGameObject = new GameObject("Root");
#if ENABLE_URP
            if (importSettings.addShadowCasters && layers.Count > 1)
                rootGameObject.AddComponent<UnityEngine.Rendering.Universal.CompositeShadowCaster2D>();
#endif
            if (importSettings.addSortingGroup && layers.Count > 1)
                rootGameObject.AddComponent<SortingGroup>();

            if (layers.Count == 1)
            {
                layerIdToGameObject.Add(layers[0].index, rootGameObject);
            }
            else
                CreateLayerHierarchy(layers, layerIdToGameObject, rootGameObject);

            for (var i = layers.Count - 1; i >= 0; --i)
            {
                var layer = layers[i];
                SetupLayerGameObject(layer, layerIdToGameObject, output.sprites, importSettings, canvasSize);

                if (layer.parentIndex == -1)
                    continue;

                var parentGo = layerIdToGameObject[layer.parentIndex];
                layerIdToGameObject[layer.index].transform.parent = parentGo.transform;
            }

            // We need the GameObjects in order to generate Animation Clips.
            // But we will only save down the GameObjects if it is requested.
            if (importSettings.generateModelPrefab)
            {
                ctx.AddObjectToAsset(rootGameObject.name, rootGameObject);
                mainAsset = rootGameObject;
            }
            else
                rootGameObject.hideFlags = HideFlags.HideAndDontSave;
        }

        static void CreateLayerHierarchy(List<Layer> layers, Dictionary<int, GameObject> layerIdToGameObject, GameObject root)
        {
            for (var i = layers.Count - 1; i >= 0; --i)
            {
                var layer = layers[i];
                var go = new GameObject(layer.name);
                go.transform.parent = root.transform;
                go.transform.localRotation = Quaternion.identity;

                layerIdToGameObject.Add(layer.index, go);
            }
        }

        static void SetupLayerGameObject(
            Layer layer,
            Dictionary<int, GameObject> layerIdToGameObject,
            Sprite[] sprites,
            AsepriteImporterSettings importSettings,
            Vector2Int canvasSize)
        {
            if (layer.cells.Count == 0)
                return;

            var firstCell = layer.cells[0];
            var gameObject = layerIdToGameObject[layer.index];
            var sprite = Array.Find(sprites, x => x.GetSpriteID() == firstCell.spriteId);

            var sr = gameObject.AddComponent<SpriteRenderer>();
            sr.sprite = sprite;
            sr.sortingOrder = layer.index + firstCell.additiveSortOrder;

#if ENABLE_URP
            if (importSettings.addShadowCasters)
                gameObject.AddComponent<UnityEngine.Rendering.Universal.ShadowCaster2D>();
#endif

            if (importSettings.defaultPivotSpace == PivotSpaces.Canvas)
                gameObject.transform.localPosition = Vector3.zero;
            else
            {
                var cellRect = firstCell.cellRect;
                var position = new Vector3(cellRect.x, cellRect.y, 0f);

                var pivot = sprite.pivot;
                position.x += pivot.x;
                position.y += pivot.y;

                var globalPivot = ImportUtilities.PivotAlignmentToVector(importSettings.defaultPivotAlignment);
                position.x -= (canvasSize.x * globalPivot.x);
                position.y -= (canvasSize.y * globalPivot.y);

                position.x /= sprite.pixelsPerUnit;
                position.y /= sprite.pixelsPerUnit;

                gameObject.transform.localPosition = position;
            }
        }
    }
}
