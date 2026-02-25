// 2After
using System;

public class Enemy {
    public event Action OnDeath;

    public void Die() {
        System.Console.WriteLine("Enemy died");
        OnDeath?.Invoke();
    }
}