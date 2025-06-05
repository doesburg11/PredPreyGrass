using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Tilemaps;

namespace UnityEditor.Tilemaps
{
    /// <summary>
    /// Tile Template which places created Tiles into fixed positions passed with this Template
    /// </summary>
    [Serializable]
    public class PositionTileTemplate : TileTemplate
    {
        [SerializeField]
        private List<Vector3Int> m_Positions;

        /// <summary>
        /// Positions to place Sprites into Tile Palette
        /// </summary>
        public IEnumerable<Vector3Int> positions
        {
            get => m_Positions;
            set
            {
                if (m_Positions == null)
                    m_Positions = new List<Vector3Int>();
                m_Positions.Clear();
                m_Positions.AddRange(value);
            }
        }

        /// <summary>
        /// Tile Template which places created Tiles into fixed positions passed with this Template
        /// </summary>
        /// <param name="positions">Positions to place Sprites into Tile Palette</param>
        public PositionTileTemplate(IEnumerable<Vector3Int> positions)
        {
            m_Positions = new List<Vector3Int>();
            m_Positions.AddRange(positions);
        }

        private static IEnumerable<(T1, T2)> MultipleEnumerate<T1, T2>(IEnumerable<T1> t1s, IEnumerable<T2> t2s)
        {
            using IEnumerator<T1> enum1 = t1s.GetEnumerator();
            using IEnumerator<T2> enum2 = t2s.GetEnumerator();
            while (enum1.MoveNext() && enum2.MoveNext())
                yield return (enum1.Current, enum2.Current);
        }

        /// <summary>
        /// Creates a List of TileBase Assets from Texture2D and Sprites with placement
        /// data onto a Tile Palette.
        /// </summary>
        /// <param name="texture2D">Texture2D to generate Tile Assets from.</param>
        /// <param name="sprites">Sprites to generate Tile Assets from. Each Sprite will be created as a Tile asset and mapped to a position in order.</param>
        /// <param name="tilesToAdd">Tile Assets and placement data to generate.</param>
        public override void CreateTileAssets(Texture2D texture2D, IEnumerable<Sprite> sprites, ref List<TileChangeData> tilesToAdd)
        {
            var createTileMethod = GridPaintActiveTargetsPreferences.GetCreateTileFromPaletteUsingPreferences();
            if (createTileMethod == null)
                return;

            var i = 0;
            var uniqueNames = new HashSet<string>();
            var tileMap = new Dictionary<Sprite, TileBase>();
            foreach (var (textureSprite, position) in MultipleEnumerate(sprites, positions))
            {
                if (!tileMap.TryGetValue(textureSprite, out TileBase tile))
                {
                    tile = createTileMethod.Invoke(null, new object[] { textureSprite }) as TileBase;
                    tileMap.Add(textureSprite, tile);
                }
                if (tile == null)
                    continue;

                var tileName = tile.name;
                if (string.IsNullOrEmpty(tileName) || uniqueNames.Contains(tileName))
                {
                    tileName = TileDragAndDrop.GenerateUniqueNameForNamelessSprite(textureSprite, uniqueNames, ref i);
                    tile.name = tileName;
                }
                uniqueNames.Add(tileName);

                tilesToAdd.Add(new TileChangeData()
                {
                    position = position,
                    tile = tile,
                    transform = Matrix4x4.identity,
                    color = Color.white
                });
            }
        }
    }
}
