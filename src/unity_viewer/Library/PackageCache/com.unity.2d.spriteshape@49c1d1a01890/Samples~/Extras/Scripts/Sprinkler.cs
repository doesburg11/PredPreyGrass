using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.U2D;

namespace SpriteShapeExtras
{
    public class Sprinkler : MonoBehaviour
    {

        public GameObject m_Prefab;
        public bool m_UseNormals = false;
        public int m_Instances = 10;

        // Use this for initialization. Plant the Prefabs on Startup
        void Start ()
        {
            SpriteShapeController ssc = GetComponent<SpriteShapeController>();
            for (int i = 1; i < m_Instances; ++i)
            {
                var go = GameObject.Instantiate(m_Prefab);
                var op = go.AddComponent<SpriteShapeObjectPlacement>();
                op.spriteShapeController = ssc;
                op.setNormal = m_UseNormals;
                op.startPoint = 0;
                op.endPoint = ssc.spline.GetPointCount();
                op.mode = SpriteShapeObjectPlacementMode.Manual;
                op.ratio = Random.Range(0.0f, 1.0f);
            }
        }
    }
}