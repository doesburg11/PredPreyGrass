using System;
using System.Linq;
using UnityEditor.Presets;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.U2D;

namespace UnityEditor.U2D.SpriteShape
{
    internal class GameObjectCreation
    {
        const int k_MenuPriority = 4;
        [MenuItem("GameObject/2D Object/Sprite Shape/Open Shape", false, k_MenuPriority)]
        static void MenuItem_GameObject2DObjectSpriteShapeOpenShape(MenuCommand menuCommand)
        {
            var asset = AssetDatabase.LoadAssetAtPath<GameObject>("Packages/com.unity.2d.spriteshape/Editor/ObjectMenuCreation/DefaultAssets/Sprite Shapes/Open Sprite Shape.prefab") as GameObject;
            var preset = new PresetType(asset.GetComponent<SpriteShapeController>());
            var defaults = Preset.GetDefaultPresetsForType(preset).Count(x => x.enabled);

            var go = CreateGameObject("Open Sprite Shape", menuCommand, new []{typeof(SpriteShapeController)});
            go.GetComponent<SpriteShapeController>().spline.isOpenEnded = true;

            if (defaults == 0)
                EditorUtility.CopySerialized(asset.GetComponent<SpriteShapeController>(), go.GetComponent<SpriteShapeController>());
        }

        [MenuItem("GameObject/2D Object/Sprite Shape/Closed Shape", false, k_MenuPriority)]
        static void MenuItem_GameObject2DObjectSpriteShapeClosedShape(MenuCommand menuCommand)
        {
            var asset = AssetDatabase.LoadAssetAtPath<GameObject>("Packages/com.unity.2d.spriteshape/Editor/ObjectMenuCreation/DefaultAssets/Sprite Shapes/Closed Sprite Shape.prefab") as GameObject;
            var preset = new PresetType(asset.GetComponent<SpriteShapeController>());
            var defaults = Preset.GetDefaultPresetsForType(preset).Count(x => x.enabled);

            var go = CreateGameObject("Closed Sprite Shape", menuCommand, new []{typeof(SpriteShapeController)});
            go.GetComponent<SpriteShapeController>().spline.isOpenEnded = false;

            if (defaults == 0)
                EditorUtility.CopySerialized(asset.GetComponent<SpriteShapeController>(), go.GetComponent<SpriteShapeController>());
        }

        internal static GameObject CreateGameObject(string name, MenuCommand menuCommand, params Type[] components)
        {
            var parent = menuCommand.context as GameObject;
            var newGO = ObjectFactory.CreateGameObject(name, components);
            newGO.name = name;
            Selection.activeObject = newGO;
            Place(newGO, parent);

            Undo.RegisterCreatedObjectUndo(newGO, string.Format("Create {0}", name));
            return newGO;
        }
        
        internal static void Place(GameObject go, GameObject parentTransform)
        {
            if (parentTransform != null)
            {
                var transform = go.transform;
                Undo.SetTransformParent(transform, parentTransform.transform, "Reparenting");
                transform.localPosition = Vector3.zero;
                transform.localRotation = Quaternion.identity;
                transform.localScale = Vector3.one;
                go.layer = parentTransform.gameObject.layer;

                if (parentTransform.GetComponent<RectTransform>())
                    ObjectFactory.AddComponent<RectTransform>(go);
            }
            else
            {
                PlaceGameObjectInFrontOfSceneView(go);

                StageUtility.PlaceGameObjectInCurrentStage(go); // may change parent
            }

            // Only at this point do we know the actual parent of the object and can modify its name accordingly.
            GameObjectUtility.EnsureUniqueNameForSibling(go);
            Undo.SetCurrentGroupName("Create " + go.name);
            Selection.activeGameObject = go;
            if (EditorSettings.defaultBehaviorMode == EditorBehaviorMode.Mode2D)
            {
                var position = go.transform.position;
                position.z = 0;
                go.transform.position = position;
            }
        }
        
        internal static void PlaceGameObjectInFrontOfSceneView(GameObject go)
        {
            var view = SceneView.lastActiveSceneView;
            if (view != null)
            {
                view.MoveToView(go.transform);
            }
        }
    }
}

