using System;
using System.Collections;
using System.Collections.Generic;

namespace UnityEngine.TestTools
{
    internal class EnumeratorHelper
    {
        public static bool IsRunningNestedEnumerator => enumeratorStack.Count > 0;

        private static IEnumerator currentEnumerator;
        private static Stack<IEnumerator> enumeratorStack = new Stack<IEnumerator>();

        /// <summary>
        /// This method executes a given enumerator and all nested enumerators.
        /// If any resuming (setting of pc) is needed, it needs to be done before being passed to this method.
        /// </summary>
        public static IEnumerator UnpackNestedEnumerators(IEnumerator testEnumerator)
        {
            if (testEnumerator == null)
            {
                throw new ArgumentNullException(nameof(testEnumerator));
            }

            currentEnumerator = testEnumerator;
            enumeratorStack.Clear();

            return ProgressOnEnumerator();
        }

        private static IEnumerator ProgressOnEnumerator()
        {
            while (true)
            {
                if (!currentEnumerator.MoveNext())
                {
                    if (enumeratorStack.Count == 0)
                    {
                        yield break;
                    }
                    currentEnumerator = enumeratorStack.Pop();
                    continue;
                }
                
                if (currentEnumerator.Current is IEnumerator nestedEnumerator)
                {
                    enumeratorStack.Push(currentEnumerator);
                    currentEnumerator = nestedEnumerator;
                }
                else
                {
                    yield return currentEnumerator.Current;
                }
            }
        }
        
        public static void SetEnumeratorPC(int pc)
        {
            if (currentEnumerator == null)
            {
                throw new Exception("No enumerator is currently running.");
            }
            
            if (IsRunningNestedEnumerator)
            {
                throw new Exception("Cannot set the enumerator PC while running nested enumerators.");
            }
            
            ActivePcHelper.SetEnumeratorPC(currentEnumerator, pc);
        }

        public static int GetEnumeratorPC()
        {
            if (currentEnumerator == null)
            {
                throw new Exception("No enumerator is currently running.");
            }

            if (IsRunningNestedEnumerator)
            {
                // Restrict the getting of PC, as it will not reflect what is currently running;
                throw new Exception("Cannot get the enumerator PC while running nested enumerators.");
            }

            return ActivePcHelper.GetEnumeratorPC(currentEnumerator);
        }

        private static TestCommandPcHelper pcHelper;
        internal static TestCommandPcHelper ActivePcHelper
        {
            get
            {
                if (pcHelper == null)
                {
                    pcHelper = new TestCommandPcHelper();
                }

                return pcHelper;
            }
            set
            {
                pcHelper = value;
            }
        }
    }
}
