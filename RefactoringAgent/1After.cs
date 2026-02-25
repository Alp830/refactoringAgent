// 1After
using System;

public class Player {
    public event Action OnJump;

    public void Jump() {
        Console.WriteLine("Jump");
        OnJump?.Invoke();
    }
}