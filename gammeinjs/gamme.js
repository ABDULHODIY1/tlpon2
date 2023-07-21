var board;
var playerO = 'O';
varplayerX = 'X';
var currPlayer = playerO;
var gameOver = false;


window.onload = function () {
    SetGame();
}
function SetGame() {
    board = [
        [' ', ' ', ' '],
        [' ', ' ', ' '],
        [' ', ' ', ' ']
    ]

    for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
            let tile = document.createElement("div");
            tile.id = r.toString() + "-" + c.toString();
            tile.classList.add("tile");
            if (r == 0 || r == 1) {
                tile.classList.add("horizantal-line");
            }
            if (c == 0 || c == 1) {
                tile.classList.add("vertical-line");
            }
            document.getElementById("board").append(tile);
        }
    }

}