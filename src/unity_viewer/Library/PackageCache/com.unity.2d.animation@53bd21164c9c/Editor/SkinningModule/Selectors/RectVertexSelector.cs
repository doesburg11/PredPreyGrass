using UnityEngine;

namespace UnityEditor.U2D.Animation
{
    internal class RectVertexSelector : IRectSelector<int>
    {
        public ISelection<int> selection { get; set; }
        public BaseSpriteMeshData spriteMeshData { get; set; }
        public Rect rect { get; set; }

        public void Select()
        {
            for (var i = 0; i < spriteMeshData.vertexCount; i++)
            {
                if (rect.Contains(spriteMeshData.vertices[i], true))
                    selection.Select(i, true);
            }
        }
    }
}
