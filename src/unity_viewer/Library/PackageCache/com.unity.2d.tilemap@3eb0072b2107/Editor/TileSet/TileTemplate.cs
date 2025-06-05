using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Tilemaps;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// Abstract Class used as a Template to create Tile Assets from Texture2D and Sprites.
    /// </summary>
    public abstract class TileTemplate : ScriptableObject
    {
        /// <summary>
        /// Creates a List of TileBase Assets from Texture2D and Sprites with placement
        /// data onto a Tile Palette.
        /// </summary>
        /// <param name="texture2D">Texture2D to generate Tile Assets from.</param>
        /// <param name="sprites">Sprites to generate Tile Assets from.</param>
        /// <param name="tilesToAdd">Tile Assets and placement data to generate.</param>
        public abstract void CreateTileAssets(Texture2D texture2D
            , IEnumerable<Sprite> sprites
            , ref List<TileChangeData> tilesToAdd);
    }
}
