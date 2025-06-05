using System;
using UnityEngine;

namespace PlayModeTests_9s
{
    public class MovementScript : MonoBehaviour
    {
        public Vector3 velocity;

        private void Update()
        {
            transform.position += velocity * Time.deltaTime;
        }
    }
}