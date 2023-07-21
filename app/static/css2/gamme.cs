using System;

class Player
{
    public string Name { get; set; }
    public string Weapon { get; set; }

    public Player(string name, string weapon)
    {
        Name = name;
        Weapon = weapon;
    }

    public void Attack(Player target)
    {
        Console.WriteLine($"{Name} attacks {target.Name} with {Weapon}!");
    }
}

class Program
{
    static void Main()
    {
        // O'yinchilarni yaratish
        Player player1 = new Player("Player 1", "Sword");
        Player player2 = new Player("Player 2", "Bow");

        // O'yinda harakatlanish
        player1.Attack(player2);
        player2.Attack(player1);
    }
}
