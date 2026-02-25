// 3After
using System;

public class Door {
    public event Action OnOpen;

    public void Open() {
        System.Console.WriteLine("Door opened");
        OnOpen?.Invoke();
    }
}