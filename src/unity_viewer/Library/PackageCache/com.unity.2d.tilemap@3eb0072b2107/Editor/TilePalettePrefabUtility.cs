using System.Collections.Generic;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace UnityEditor.Tilemaps
{
    internal class TilePalettePrefabUtility
    {
        internal static bool IsObjectPrefabInstance(Object target)
        {
            return target != null && PrefabUtility.IsPartOfRegularPrefab(target);
        }

        internal static GameObject FindPrefabInstanceEquivalent(GameObject prefabInstance, GameObject prefabTarget)
        {
            var prefabRoot = prefabTarget.transform.root.gameObject;
            var currentTransform = prefabTarget.transform;
            var reverseTransformOrder = new Stack<int>();
            while (currentTransform != prefabRoot.transform && currentTransform.parent != null)
            {
                var parentTransform = currentTransform.parent;
                for (int i = 0; i < parentTransform.childCount; ++i)
                {
                    if (currentTransform == parentTransform.GetChild(i))
                    {
                        reverseTransformOrder.Push(i);
                        break;
                    }
                }
                currentTransform = currentTransform.parent;
            }

            currentTransform = prefabInstance.transform;
            while (reverseTransformOrder.Count > 0)
            {
                var childIndex = reverseTransformOrder.Pop();
                if (childIndex >= currentTransform.childCount)
                    return null;
                currentTransform = currentTransform.GetChild(childIndex);
            }
            return currentTransform.gameObject;
        }

        internal static void GoToPrefabMode(GameObject target)
        {
            var prefabObject = PrefabUtility.GetCorrespondingObjectFromSource(target);
            var assetPath = AssetDatabase.GetAssetPath(prefabObject);
            var stage = PrefabStageUtility.OpenPrefab(assetPath);
            var prefabInstance = stage.prefabContentsRoot;
            var prefabTarget = FindPrefabInstanceEquivalent(prefabInstance, prefabObject);
            if (prefabTarget != null)
            {
                GridPaintingState.scenePaintTarget = prefabTarget;
            }
        }
    }
}
