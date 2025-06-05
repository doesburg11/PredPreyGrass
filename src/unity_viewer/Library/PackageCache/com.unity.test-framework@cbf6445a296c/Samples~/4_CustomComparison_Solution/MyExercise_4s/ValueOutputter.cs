using UnityEngine;

namespace MyExercise_4s
{
    public class ValueOutputter
    {
        public Vector3 GetVector3()
        {
            return new Vector3(10f + (1f/3f), 3f, 9f + (2f/3f));
        }
        
        public Quaternion GetQuaternion()
        {
            return new Quaternion(10f, 0f, 7f + (1f/3f), 0f);
        }
        
        public float GetFloat()
        {
            return 19f + (1f/3f);
        }
    }
}