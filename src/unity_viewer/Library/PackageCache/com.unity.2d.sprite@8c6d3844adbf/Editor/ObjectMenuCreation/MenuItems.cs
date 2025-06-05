using UnityEngine;
using UnityEngine.U2D;
using UnityEditor;

namespace UnityEditor.U2D
{
    static internal class MenuItems
    {
        enum SpriteAssetMenuPriority : int
        {
            Triangle = 1,
            Square,
            Circle,
            Capsule,
            IsometricDiamond,
            HexagonFlatTop,
            HexagonPointTop,
            Sliced9
        }

        enum SpriteAtlasAssetMenuPriority : int
        {
            SpriteAtlas = SpriteAssetMenuPriority.Triangle + 11
        }

        enum SpriteGameObjectMenuPriority : int
        {
            Triangle = 1,
            Square,
            Circle,
            Capsule,
            IsometricDiamond,
            HexagonFlatTop,
            HexagonPointTop,
            Sliced9
        }

        enum PhysicsGameObjectMenuPriority : int
        {
            StaticSprite = 2,
            DynamicSprite
        }

        enum SpriteMaskGameObjectMenuPriority : int
        {
            SpriteMask = 6
        }

        [MenuItem("Assets/Create/2D/Sprites/Triangle", priority = (int)SpriteAssetMenuPriority.Triangle)]
        static void AssetsCreateSpritesTriangle(MenuCommand menuCommand)
        {
            ItemCreationUtility.CreateAssetObjectFromTemplate<Texture2D>("Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/Triangle.png");
        }

        [MenuItem("Assets/Create/2D/Sprites/Square", priority = (int)SpriteAssetMenuPriority.Square)]
        static void AssetsCreateSpritesSquare(MenuCommand menuCommand)
        {
            ItemCreationUtility.CreateAssetObjectFromTemplate<Texture2D>("Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/Square.png");
        }

        [MenuItem("Assets/Create/2D/Sprites/Circle", priority = (int)SpriteAssetMenuPriority.Circle)]
        static void AssetsCreateSpritesCircle(MenuCommand menuCommand)
        {
            ItemCreationUtility.CreateAssetObjectFromTemplate<Texture2D>("Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/Circle.png");
        }

        [MenuItem("Assets/Create/2D/Sprites/Capsule", priority = (int)SpriteAssetMenuPriority.Capsule)]
        static void AssetsCreateSpritesCapsule(MenuCommand menuCommand)
        {
            ItemCreationUtility.CreateAssetObjectFromTemplate<Texture2D>("Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/Capsule.png");
        }

        [MenuItem("Assets/Create/2D/Sprites/Isometric Diamond", priority = (int)SpriteAssetMenuPriority.IsometricDiamond)]
        static void AssetsCreateSpritesIsometricDiamond(MenuCommand menuCommand)
        {
            ItemCreationUtility.CreateAssetObjectFromTemplate<Texture2D>("Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/IsometricDiamond.png");
        }

        [MenuItem("Assets/Create/2D/Sprites/Hexagon Flat Top", priority = (int)SpriteAssetMenuPriority.HexagonFlatTop)]
        static void AssetsCreateSpritesHexagonFlatTop(MenuCommand menuCommand)
        {
            ItemCreationUtility.CreateAssetObjectFromTemplate<Texture2D>("Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/HexagonFlatTop.png");
        }

        [MenuItem("Assets/Create/2D/Sprites/Hexagon Point Top", priority = (int)SpriteAssetMenuPriority.HexagonPointTop)]
        static void AssetsCreateSpritesHexagonPointTop(MenuCommand menuCommand)
        {
            ItemCreationUtility.CreateAssetObjectFromTemplate<Texture2D>("Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/HexagonPointTop.png");
        }

        [MenuItem("Assets/Create/2D/Sprites/9-Sliced", priority = (int)SpriteAssetMenuPriority.Sliced9)]
        static void AssetsCreateSprites9Sliced(MenuCommand menuCommand)
        {
            ItemCreationUtility.CreateAssetObjectFromTemplate<Texture2D>("Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/9Sliced.png");
        }

        internal class DoCreateSpriteAtlas : ProjectWindowCallback.EndNameEditAction
        {
            public int sides;
            public override void Action(int instanceId, string pathName, string resourceFile)
            {
                var spriteAtlasAsset = new SpriteAtlasAsset();

                UnityEditorInternal.InternalEditorUtility.SaveToSerializedFileAndForget(new Object[] { spriteAtlasAsset }, pathName, true);
                AssetDatabase.Refresh(ImportAssetOptions.ForceUpdate);
            }
        }

        static private void CreateSpriteAtlas()
        {
            var icon = EditorGUIUtility.IconContent<SpriteAtlasAsset>().image as Texture2D;
            DoCreateSpriteAtlas action = ScriptableObject.CreateInstance<DoCreateSpriteAtlas>();
            ProjectWindowUtil.StartNameEditingIfProjectWindowExists(0, action, "New Sprite Atlas.spriteatlasv2", icon, null);
        }

        [MenuItem("Assets/Create/2D/Sprite Atlas", priority = (int)SpriteAtlasAssetMenuPriority.SpriteAtlas)]
        static void AssetsCreateSpriteAtlas(MenuCommand menuCommand)
        {
            if (EditorSettings.spritePackerMode == SpritePackerMode.SpriteAtlasV2 || EditorSettings.spritePackerMode == SpritePackerMode.SpriteAtlasV2Build)
                CreateSpriteAtlas();
            else
                ItemCreationUtility.CreateAssetObject<SpriteAtlas>("New Sprite Atlas.spriteatlas");
        }

        static GameObject CreateSpriteRendererGameObject(string name,  string spritePath, MenuCommand menuCommand)
        {
            var go = ItemCreationUtility.CreateGameObject(name, menuCommand, new[] {typeof(SpriteRenderer)});
            var sr = go.GetComponent<SpriteRenderer>();
            sr.sprite = AssetDatabase.LoadAssetAtPath<Sprite>(spritePath);
            return go;
        }

        [MenuItem("GameObject/2D Object/Sprites/Triangle", priority = (int)SpriteAssetMenuPriority.Triangle)]
        static void GameObjectCreateSpritesTriangle(MenuCommand menuCommand)
        {
            CreateSpriteRendererGameObject("Triangle", "Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/Triangle.png", menuCommand);
        }

        [MenuItem("GameObject/2D Object/Sprites/Square", priority = (int)SpriteAssetMenuPriority.Square)]
        static void GameObjectCreateSpritesSquare(MenuCommand menuCommand)
        {
            CreateSpriteRendererGameObject("Square", "Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/Square.png", menuCommand);
        }

        [MenuItem("GameObject/2D Object/Sprites/Circle", priority = (int)SpriteAssetMenuPriority.Circle)]
        static void GameObjectCreateSpritesCircle(MenuCommand menuCommand)
        {
            CreateSpriteRendererGameObject("Circle", "Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/Circle.png", menuCommand);
        }

        [MenuItem("GameObject/2D Object/Sprites/Capsule", priority = (int)SpriteAssetMenuPriority.Capsule)]
        static void GameObjectCreateSpritesCapsule(MenuCommand menuCommand)
        {
            CreateSpriteRendererGameObject("Capsule", "Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/Capsule.png", menuCommand);
        }

        [MenuItem("GameObject/2D Object/Sprites/Isometric Diamond", priority = (int)SpriteAssetMenuPriority.IsometricDiamond)]
        static void GameObjectCreateSpritesIsometricDiamond(MenuCommand menuCommand)
        {
            CreateSpriteRendererGameObject("Isometric Diamond", "Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/IsometricDiamond.png", menuCommand);
        }

        [MenuItem("GameObject/2D Object/Sprites/Hexagon Flat Top", priority = (int)SpriteAssetMenuPriority.HexagonFlatTop)]
        static void GameObjectCreateSpritesHexagonFlatTop(MenuCommand menuCommand)
        {
            CreateSpriteRendererGameObject("Hexagon Flat Top", "Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/HexagonFlatTop.png", menuCommand);
        }

        [MenuItem("GameObject/2D Object/Sprites/Hexagon Point Top", priority = (int)SpriteAssetMenuPriority.HexagonPointTop)]
        static void GameObjectCreateSpritesHexagonPointedTop(MenuCommand menuCommand)
        {
            CreateSpriteRendererGameObject("Hexagon Point Top", "Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/HexagonPointTop.png", menuCommand);
        }

        [MenuItem("GameObject/2D Object/Sprites/9-Sliced", priority = (int)SpriteAssetMenuPriority.Sliced9)]
        static void GameObjectCreateSprites9Sliced(MenuCommand menuCommand)
        {
            var go = CreateSpriteRendererGameObject("9-Sliced", "Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/9Sliced.png", menuCommand);
            var sr = go.GetComponent<SpriteRenderer>();
            if (sr.drawMode == SpriteDrawMode.Simple)
            {
                sr.drawMode = SpriteDrawMode.Tiled;
                sr.tileMode = SpriteTileMode.Continuous;
            }
        }

        [MenuItem("GameObject/2D Object/Physics/Static Sprite", priority = (int)PhysicsGameObjectMenuPriority.StaticSprite)]
        static void GameObjectCreatePhysicsStaticSprite(MenuCommand menuCommand)
        {
            var go = ItemCreationUtility.CreateGameObject("Static Sprite", menuCommand, new[] {typeof(SpriteRenderer), typeof(BoxCollider2D), typeof(Rigidbody2D)});
            var sr = go.GetComponent<SpriteRenderer>();
            if (sr.sprite == null)
                sr.sprite = AssetDatabase.LoadAssetAtPath<Sprite>(
                    "Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/Square.png");
            var rigidBody = go.GetComponent<Rigidbody2D>();
            rigidBody.bodyType = RigidbodyType2D.Static;
            var boxCollider2D = go.GetComponent<BoxCollider2D>();
            boxCollider2D.size = sr.sprite.rect.size / sr.sprite.pixelsPerUnit;
        }

        [MenuItem("GameObject/2D Object/Physics/Dynamic Sprite", priority = (int)PhysicsGameObjectMenuPriority.DynamicSprite)]
        static void GameObjectCreatePhysicsDynamicSprite(MenuCommand menuCommand)
        {
            var go = ItemCreationUtility.CreateGameObject("Dynamic Sprite", menuCommand, new[] {typeof(SpriteRenderer), typeof(CircleCollider2D), typeof(Rigidbody2D)});
            var sr = go.GetComponent<SpriteRenderer>();
            if (sr.sprite == null)
                sr.sprite = AssetDatabase.LoadAssetAtPath<Sprite>(
                    "Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/v2/Circle.png");
            var rigidBody = go.GetComponent<Rigidbody2D>();
            rigidBody.bodyType = RigidbodyType2D.Dynamic;
        }

        [MenuItem("GameObject/2D Object/Sprite Mask", priority = (int)SpriteMaskGameObjectMenuPriority.SpriteMask)]
        static void GameObjectCreateSpriteMask(MenuCommand menuCommand)
        {
            var go = ItemCreationUtility.CreateGameObject("Sprite Mask", menuCommand, new[] {typeof(SpriteMask)});
            go.GetComponent<SpriteMask>().sprite = AssetDatabase.LoadAssetAtPath<Sprite>(
                "Packages/com.unity.2d.sprite/Editor/ObjectMenuCreation/DefaultAssets/Textures/CircleMask.png");
        }
    }
}
