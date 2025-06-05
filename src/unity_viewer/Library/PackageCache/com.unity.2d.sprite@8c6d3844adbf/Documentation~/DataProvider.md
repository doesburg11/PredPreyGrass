# Sprite Editor Data Provider API

By using the Sprite Editor Data Provider API, the user can add, change and remove Sprite data in a custom importer or editor tool. Refer to the code examples below to see how the API is applied.

**Important:** Some of the following examples contains an additional section of code which is needed if you are using Unity 2021.2 or newer. If you are using Unity 2021.1 or older, you should remove the indicated section to ensure the code compiles properly.

## How to get ISpriteEditorDataProvider instances

The following examples show you how to use the API to get each respective instance.

### Importer

```C#
using UnityEditor;
using UnityEditor.U2D.Sprites;
using UnityEngine;

public class MyAssetPostProcessor : AssetPostprocessor
{
    private void OnPreprocessTexture()
    {
        var factory = new SpriteDataProviderFactories();
        factory.Init();
        var dataProvider = factory.GetSpriteEditorDataProviderFromObject(assetImporter);
        dataProvider.InitSpriteEditorDataProvider();

        /* Use the data provider */

        // Apply the changes made to the data provider
        dataProvider.Apply();
    }
}
```

### Texture

```C#
using UnityEditor;
using UnityEditor.U2D.Sprites;
using UnityEngine;

public static class MyCustomTool
{
    [MenuItem("Custom/Update Sprite Settings")]
    static void UpdateSettings()
    {
        foreach (var obj in Selection.objects)
        {
            if (obj is Texture2D)
            {
                var factory = new SpriteDataProviderFactories();
                factory.Init();
                var dataProvider = factory.GetSpriteEditorDataProviderFromObject(obj);
                dataProvider.InitSpriteEditorDataProvider();

                /* Use the data provider */

                // Apply the changes made to the data provider
                dataProvider.Apply();

                // Reimport the asset to have the changes applied
                var assetImporter = dataProvider.targetObject as AssetImporter;
                assetImporter.SaveAndReimport();
            }
        }
    }
}
```

## How to add Sprites

```C#
static void AddSprite(ISpriteEditorDataProvider dataProvider)
{
    // Define the new Sprite Rect
    var newSprite = new SpriteRect()
    {
        name = "MyNewSprite",
        spriteID = GUID.Generate(),
        rect = new Rect(0, 0, 32, 32)
    };
    // Add the Sprite Rect to the list of existing Sprite Rects
    var spriteRects = dataProvider.GetSpriteRects().ToList();
    spriteRects.Add(newSprite);

    // Write the updated data back to the data provider
    dataProvider.SetSpriteRects(spriteRects.ToArray());

    // Note: This section is only for Unity 2021.2 and newer
    // Register the new Sprite Rect's name and GUID with the ISpriteNameFileIdDataProvider
    var spriteNameFileIdDataProvider = dataProvider.GetDataProvider<ISpriteNameFileIdDataProvider>();
    var nameFileIdPairs = spriteNameFileIdDataProvider.GetNameFileIdPairs().ToList();
    nameFileIdPairs.Add(new SpriteNameFileIdPair(newSprite.name, newSprite.spriteID));
    spriteNameFileIdDataProvider.SetNameFileIdPairs(nameFileIdPairs);
    // End of Unity 2021.2 and newer section

    // Apply the changes
    dataProvider.Apply();
}
```

## How to change Sprite data

```C#
static void SetPivot(ISpriteEditorDataProvider dataProvider)
{
    // Get all the existing Sprites
    var spriteRects = dataProvider.GetSpriteRects();

    // Loop over all Sprites and update the pivots
    foreach (var rect in spriteRects)
    {
        rect.pivot = new Vector2(0.1f, 0.2f);
        rect.alignment = SpriteAlignment.Custom;
    }

    // Write the updated data back to the data provider
    dataProvider.SetSpriteRects(spriteRects);

    // Apply the changes
    dataProvider.Apply();
}
```

## How to remove Sprites

```C#
static void RemoveSprite(ISpriteEditorDataProvider dataProvider, string spriteName)
{
    // Get all the existing Sprites and look for the Sprite with the selected name
    var spriteRects = dataProvider.GetSpriteRects().ToList();
    var index = spriteRects.FindIndex(x => x.name == spriteName);

    // Remove the entry of the Sprite if found
    if (index >= 0)
        spriteRects.RemoveAt(index);

    // Write the updated data back to the data provider
    dataProvider.SetSpriteRects(spriteRects.ToArray());

    // Note: This section is only for Unity 2021.2 and newer
    // Get all the existing SpriteName & FileId pairs and look for the Sprite with the selected name
    var spriteNameFileIdDataProvider = dataProvider.GetDataProvider<ISpriteNameFileIdDataProvider>();
    var nameFileIdPairs = spriteNameFileIdDataProvider.GetNameFileIdPairs().ToList();
    index = nameFileIdPairs.FindIndex(x => x.name == spriteName);

    // Remove the entry of the Sprite if found
    if (index >= 0)
        nameFileIdPairs.RemoveAt(index);

    spriteNameFileIdDataProvider.SetNameFileIdPairs(nameFileIdPairs);
    // End of Unity 2021.2 and newer section

    // Apply the changes
    dataProvider.Apply();
}
```

## How to update Outline data

```C#
static void SetOutline(ISpriteEditorDataProvider dataProvider)
{
    // Get the ISpriteOutlineDataProvider
    var outlineDataProvider = dataProvider.GetDataProvider<ISpriteOutlineDataProvider>();

    // Loop over all Sprites and set their outline to a quad
    var spriteRects = dataProvider.GetSpriteRects();
    foreach (var spriteRect in spriteRects)
    {
        var halfWidth = spriteRect.rect.width / 2f;
        var halfHeight = spriteRect.rect.height / 2f;

        var quadOutline = new Vector2[4]
        {
            new Vector2(-halfWidth, -halfHeight),
            new Vector2(-halfWidth, halfHeight),
            new Vector2(halfWidth, halfHeight),
            new Vector2(halfWidth, -halfHeight)
        };

        var outlines = new List<Vector2[]>();
        outlines.Add(quadOutline);

        var spriteGuid = spriteRect.spriteID;
        outlineDataProvider.SetOutlines(spriteGuid, outlines);
    }

    // Apply the changes
    dataProvider.Apply();
}
```

## Additional resources

- Full list of other available data providers is available in the package's [Scripting API section](xref:UnityEditor.U2D.Sprites).
