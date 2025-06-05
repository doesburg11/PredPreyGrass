using UnityEngine;
using UnityEngine.U2D.Animation;

namespace UnityEditor.U2D.Animation
{
    [InitializeOnLoad]
    internal static class SpriteLibraryAssetDragAndDrop
    {
        const string k_UndoableCreate = "Create new Sprite Library Object";
        const string k_UndoableAdd = "Add Sprite Library";

        static SpriteLibraryAssetDragAndDrop()
        {
            DragAndDrop.AddDropHandler(HandleDropInspector);
            DragAndDrop.AddDropHandler(HandleDropHierarchy);
            DragAndDrop.AddDropHandler(HandleDropScene);
        }

        static DragAndDropVisualMode HandleDropInspector(Object[] targets, bool perform)
        {
            return HandleDropInspectorInternal(DragAndDrop.objectReferences, targets, perform);
        }

        static DragAndDropVisualMode HandleDropHierarchy(int dropTargetInstanceID, HierarchyDropFlags dropMode, Transform parentForDraggedObjects, bool perform)
        {
            return HandleDropHierarchyInternal(DragAndDrop.objectReferences, dropTargetInstanceID, dropMode, perform);
        }

        static DragAndDropVisualMode HandleDropScene(Object dropUpon, Vector3 worldPosition, Vector2 viewportPosition, Transform parentForDraggedObjects, bool perform)
        {
            return HandleDropSceneInternal(DragAndDrop.objectReferences, dropUpon, worldPosition, perform);
        }

        internal static DragAndDropVisualMode HandleDropInspectorInternal(Object[] draggedObjects, Object[] targets, bool perform)
        {
            var spriteLibraryAsset = GetSpriteLibraryAsset(draggedObjects);
            if (spriteLibraryAsset == null)
                return DragAndDropVisualMode.None;

            DragAndDrop.AcceptDrag();
            if (perform)
            {
                for (var i = 0; i < targets.Length; i++)
                {
                    if (targets[i] is GameObject targetGo)
                        AddSpriteLibraryToObject(targetGo, spriteLibraryAsset);
                }
            }

            return DragAndDropVisualMode.Copy;
        }

        internal static DragAndDropVisualMode HandleDropHierarchyInternal(Object[] draggedObjects, int dropTargetInstanceID, HierarchyDropFlags dropMode, bool perform)
        {
            var spriteLibraryAsset = GetSpriteLibraryAsset(draggedObjects);
            if (spriteLibraryAsset == null)
                return DragAndDropVisualMode.None;

            var dropUpon = EditorUtility.InstanceIDToObject(dropTargetInstanceID);
            if (dropUpon == null || dropMode == HierarchyDropFlags.DropBetween)
            {
                DragAndDrop.AcceptDrag();
                if (perform)
                    CreateSpriteLibraryObject(spriteLibraryAsset, Vector3.zero);

                return DragAndDropVisualMode.Copy;
            }

            if (dropUpon is GameObject targetGo)
            {
                DragAndDrop.AcceptDrag();
                if (perform)
                    AddSpriteLibraryToObject(targetGo, spriteLibraryAsset);

                return DragAndDropVisualMode.Link;
            }

            return DragAndDropVisualMode.None;
        }

        internal static DragAndDropVisualMode HandleDropSceneInternal(Object[] draggedObjects, Object dropUpon, Vector3 worldPosition, bool perform)
        {
            var spriteLibraryAsset = GetSpriteLibraryAsset(draggedObjects);
            if (spriteLibraryAsset == null)
                return DragAndDropVisualMode.None;

            DragAndDrop.AcceptDrag();

            if (dropUpon is GameObject targetGo)
            {
                if (perform)
                    AddSpriteLibraryToObject(targetGo, spriteLibraryAsset);

                return DragAndDropVisualMode.Link;
            }

            if (perform)
                CreateSpriteLibraryObject(spriteLibraryAsset, worldPosition);

            return DragAndDropVisualMode.Copy;
        }

        internal static SpriteLibraryAsset GetSpriteLibraryAsset(Object[] objectReferences)
        {
            for (var i = 0; i < objectReferences.Length; i++)
            {
                var draggedObject = objectReferences[i];
                if (draggedObject is SpriteLibraryAsset spriteLibraryAsset)
                    return spriteLibraryAsset;
            }

            return null;
        }


        internal static void AddSpriteLibraryToObject(GameObject targetGo, SpriteLibraryAsset spriteLibraryAsset)
        {
            Undo.RegisterFullObjectHierarchyUndo(targetGo, k_UndoableAdd);
            var spriteLibraryComponent = targetGo.GetComponent<SpriteLibrary>();
            if (spriteLibraryComponent == null)
                spriteLibraryComponent = targetGo.AddComponent<SpriteLibrary>();
            spriteLibraryComponent.spriteLibraryAsset = spriteLibraryAsset;

            Selection.objects = new Object[] { targetGo };
        }

        internal static void CreateSpriteLibraryObject(SpriteLibraryAsset spriteLibraryAsset, Vector3 position)
        {
            var newSpriteLibraryGameObject = new GameObject(spriteLibraryAsset.name);
            var transform = newSpriteLibraryGameObject.transform;
            transform.position = position;
            var spriteLibraryComponent = newSpriteLibraryGameObject.AddComponent<SpriteLibrary>();
            spriteLibraryComponent.spriteLibraryAsset = spriteLibraryAsset;
            Undo.RegisterCreatedObjectUndo(newSpriteLibraryGameObject, k_UndoableCreate);

            Selection.objects = new Object[] { newSpriteLibraryGameObject };
        }
    }
}
