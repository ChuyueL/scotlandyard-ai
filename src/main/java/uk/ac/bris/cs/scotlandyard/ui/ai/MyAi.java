package uk.ac.bris.cs.scotlandyard.ui.ai;

import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Collectors;

import javax.annotation.Nonnull;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import uk.ac.bris.cs.scotlandyard.model.*;

import uk.ac.bris.cs.scotlandyard.model.Move.*;

import uk.ac.bris.cs.scotlandyard.model.Piece.*;

import static uk.ac.bris.cs.scotlandyard.model.Board.*;
import static uk.ac.bris.cs.scotlandyard.model.ScotlandYard.*;

public class MyAi implements Ai {

	@Nonnull @Override public String name() { return "AIndrew"; }

	@Nonnull @Override public Move pickMove(
			@Nonnull Board board,
			@Nonnull AtomicBoolean terminate) {

		var moves = board.getAvailableMoves().asList();
		Move chosenMove = moves.get(new Random().nextInt(moves.size())); //Keep a random move in order to access MrX's location

		//Creating a list of the detectives in the game.
		ArrayList<Detective> detectives = new ArrayList<>();

		for (Detective detective : Detective.values()) {
			if (board.getPlayers().contains(detective)) {
				detectives.add(detective);
			}
		}

		//Creating an ImmutableMap of the detectives and their locations, in order to create the ImmutableBoard.
		HashMap<Detective, Integer> tempDetectiveLocations = new HashMap<>(); //Temporary HashMap which will be used to create an ImmutableMap
		for (Detective d : detectives) {
			tempDetectiveLocations.put(d, board.getDetectiveLocation(d).orElseThrow());
		}
		ImmutableMap<Detective, Integer> detectiveLocations = ImmutableMap.copyOf(tempDetectiveLocations);

		//Create an ImmutableMap of player tickets in order to create a list of Players. Also required for the ImmutableBoard.
		HashMap<Piece, ImmutableMap<Ticket,Integer>> tempTickets = new HashMap<>();
		ArrayList<Player> detectivePlayers = new ArrayList<>();
		for (Piece p : board.getPlayers()) {
			HashMap<Ticket, Integer> tempPlayerTickets = new HashMap<>();

			//Get counts for each ticket type for the current Piece.
			for (Ticket t : Ticket.values()) {
				tempPlayerTickets.put(t, board.getPlayerTickets(p).orElseThrow().getCount(t));
			}
			ImmutableMap<Ticket, Integer> playerTickets = ImmutableMap.copyOf(tempPlayerTickets);
			tempTickets.put(p, playerTickets);
		}
		ImmutableMap<Piece, ImmutableMap<Ticket, Integer>> tickets = ImmutableMap.copyOf(tempTickets);

		//Create a list of Players (rather than Pieces) to be used in the BoardAdapter.
		Piece mrX = MrX.MRX;
		Player MrX = new Player(mrX, tickets.get(mrX), chosenMove.source());

		for (Detective d : detectives) {
			detectivePlayers.add(new Player(d, tickets.get(d), board.getDetectiveLocation(d).get()));
		}

		ImmutableBoard immutableBoard = new ImmutableBoard(board.getSetup(), detectiveLocations, tickets, board.getMrXTravelLog(), board.getWinner(), board.getAvailableMoves());

		GameState gameState = new BoardAdapter(immutableBoard, ImmutableSet.of(MrX.piece()), MrX, detectivePlayers);

		//Creating new game tree with current game state as root.
		GameTree tree = new GameTree(gameState);

		//Run the minimax algorithm to calculate the scores for Mr X's possible moves.
		int depth = 1 + detectives.size();
		double score = minimax(tree.head, MrX.location(), detectives, depth, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY);

		chosenMove = findMaximisingMove(tree.head()); //Find the move with the highest score.

		return chosenMove;
	}

	//Finds the score of a move based on the shortest distance from the detectives to Mr X (calculated using Dijkstra's algorithm)
	double findScore(GameState gameState, int source, ArrayList<Detective> detectives) {
		Set<Integer> nodes = gameState.getSetup().graph.nodes();
		int totalDistances = 0;
		for (Detective detective : detectives) { //Find shortest distance from mrX to detective

			int destination = gameState.getDetectiveLocation(detective).get(); //Finding distance from source (mrX location) to destination (detective location)
			ArrayList<Integer> unevaluatedNodes = new ArrayList<>();
			for (Integer node : nodes) { //At the start, all of the nodes are unevaluated
				unevaluatedNodes.add(node);
			}

			//Create a distances array where distance[node key] is the shortest distance found so far from the source to that node.
			Integer maxNode = Collections.max(nodes);
			Integer[] distances = new Integer[maxNode + 1];

			for (int i = 0; i <= maxNode; i++) {
				distances[i] = Integer.MAX_VALUE; //At the start, shortest distance to each node is infinity because none have been evaluated.
			}

			distances[source] = 0; //distance to source is 0 as source is where search starts
			Integer currentNode = source;

			//Keep calculating distances until the shortest distance to the destination node is found or there are no more unevaluated nodes.
			while (!unevaluatedNodes.isEmpty() && currentNode != destination) {
				currentNode = findCurrentNode(distances); //Find node with shortest distance

				Set<Integer> neighbours = gameState.getSetup().graph.adjacentNodes(currentNode); //Find neighbours of current node

				for(Integer neighbour: neighbours) {
					Integer alt = Integer.valueOf(distances[currentNode] + 1); //Add 1 to the distance from the current node as all edges have same numerical value.

					//Check if the path through the current node is the shortest path so far.
					if (alt < distances[neighbour] && isUnevaluated(unevaluatedNodes, neighbour)) {
						distances[neighbour] = alt;
					}
				}

				unevaluatedNodes.remove(currentNode); //Remove the node from the list of unevaluated nodes.
				distances[currentNode] = Integer.MAX_VALUE; //Set the distance to the current node to infinity so that it is not evaluated again.

			}
			//Add distance to this detective to the total distance.
			//The added distance is divided by 1x10^6 because we were having trouble with the total distance overflowing.
			totalDistances += distances[destination]/1E6;

		}
		double score = totalDistances/((double) detectives.size());
		return score;

	}

	//Finds the node with the shortest distance so far, as this will be the next node to be evaluated.
	Integer findCurrentNode(Integer[] distances) {
		Integer index = 0;
		for (int i = 0; i < distances.length; i++) {
			if (distances[i] < distances[index]) {
				index = i;
			}
		}
		return index;
	}

	//Checks whether a node has already been evaluated. This is important to prevent the distances associated with
	//already visited nodes being overwritten.
	boolean isUnevaluated(ArrayList<Integer> unevaluatedNodes, Integer n) {
		for (Integer node : unevaluatedNodes) {
			if (node.equals(n)) {
				return true;
			}
		}
		return false;
	}

	//Constructs a game tree and then calculates the scores from the bottom up, using the minimax algorithm.
	double minimax(Node currentNode, int mrXLocation, ArrayList<Detective> detectives, int remainingDepth, double alpha, double beta) {
		//The current game state is the state stored in the node (the state before the next move has been made).
		GameState state = currentNode.getState();

		//If the leaves of the tree have been reached (the lowest layer which will be evaluated), calculate the score for the game state.
		if (remainingDepth == 0) {
			double score = findScore(state, mrXLocation, detectives);
			currentNode.setScore(score); //Set the score stored in the node. This is required to find the best move.
			return score;
		}
		else {
			var moves = state.getAvailableMoves().asList();
			if (!moves.isEmpty()) { //Can only get a random move if there are possible moves.
				Move tempMove = moves.get(new Random().nextInt(moves.size())); //Hold a random move to check whether it is Mr X's turn

				if (tempMove.commencedBy().isMrX()) {

					//If it's Mr X's turn, set the score of the node to the highest score of its child nodes.
					double maxScore = Double.NEGATIVE_INFINITY;
					for (Move move : moves) {
						Node child = new Node(state.advance(move));
						child.move = move;
						currentNode.addChild(child);

						// Its an array so that it can be used by the anonymous methods.
						final int[] newPlayerDest = new int[1];

						//Update the location of Mr X.
						move.visit(new Visitor<Move>() {
							@Override public Move visit(SingleMove singleMove) {
								newPlayerDest[0] = singleMove.destination;
								return singleMove;
							}

							@Override
							public Move visit(DoubleMove doubleMove) {
								newPlayerDest[0] = doubleMove.destination2;
								return doubleMove;
							}
						});

						mrXLocation = newPlayerDest[0];

						//Recursively calculate the scores for each child node.
						double childScore = minimax(child, mrXLocation, detectives, remainingDepth - 1, alpha, beta);
						maxScore = Math.max(maxScore, childScore); //Maintain the maximum score of the child nodes.
						currentNode.setScore(maxScore);

						alpha = Math.max(alpha, childScore);

						//When the highest possible score the detective (minimiser) is guaranteed becomes smaller than or equal to
						//the lowest possible score Mr X (maximiser) is guaranteed, the children of the node no longer need to be considered
						//as these moves will never be chosen.
						if (beta <= alpha) {
							break;
						}

					}
					return maxScore;

				}
				else {

					//If it's a detective's turn, set the score of the node to the lowest score of its child nodes.
					double minScore = Double.POSITIVE_INFINITY;

					//Choose a random detective for this level of the tree. This ensures that each level of the tree will only hold moves by
					//one detective.
					Piece currentDetective = tempMove.commencedBy();

					//Only include moves by the chosen detectives.
					moves = ImmutableList.copyOf(moves.stream().filter(m -> m.commencedBy() == currentDetective).collect(Collectors.toList()));
					for (Move move : moves) {
						Node child = new Node(state.advance(move));
						child.move = move;
						currentNode.addChild(child);

						double childScore = minimax(child, mrXLocation, detectives, remainingDepth - 1, alpha, beta);
						minScore = Math.min(minScore, childScore); //Maintain the minimum score of the child nodes.
						currentNode.setScore(minScore);

						beta = Math.min(beta, childScore);
						if (beta <= alpha) {
							break;
						}


					}
					return minScore;

				}
			}
			else {
				double score = findScore(state, mrXLocation, detectives);
				currentNode.setScore(score); //Set the score stored in the node. This is required to find the best move.
				return score;
			}

		}

	}

	//Finds the move from Mr X's possible moves which has the highest score, as calculated by the minimax algorithm.
	Move findMaximisingMove(Node currentNode) {
		Move chosenMove = null;
		double maxScore = 0.0;
		for (Node child : currentNode.getChildren()) {
			if (child.score > maxScore) {
				chosenMove = child.move;
			}
		}
		return chosenMove;
	}


	//Holds the root of the game tree.
	private class GameTree {
		private final Node head;

		public Node head() {
			return head;
		}

		public GameTree(GameState gameState) {
			head = new Node(gameState);
		}
	}

	//A node in the game tree.
	private class Node {
		private ArrayList<Node> children = new ArrayList<>();
		private GameState state; //The state of the game after taking the move which led to this node.
		private double score; //The score, as calculated by the minimax algorithm
		private Move move; //The move which led to this node.

		void addChild(Node node) {
			children.add(node);
		}

		private Node(GameState gameState) {
			this.state = gameState;
		}


		ArrayList<Node> getChildren() {
			return this.children;
		}

		void setScore(double score) {
			this.score = score;
		}

		GameState getState() {
			return state;
		}
	}


	//Board to GameState adapter. Required because the advance() method is used to create the minimax game tree.
	private class BoardAdapter implements GameState {
		Board board;

		private GameSetup setup;
		private Player mrX;
		private List<Player> detectives;
		private ImmutableList<LogEntry> log;
		private ImmutableSet<Move> moves;
		private ImmutableSet<Piece> winner;
		private ImmutableSet<Piece> remaining;
		private ImmutableList<Player> everyone;

		public BoardAdapter(Board board, ImmutableSet<Piece> remaining, Player mrX, List<Player> detectives) {
			this.board = board;

			this.setup = getSetup();
			this.log = getMrXTravelLog();


			this.mrX = mrX;
			this.detectives = detectives;
			this.remaining = remaining;

			ArrayList<Player> tempEveryone = new ArrayList<>();
			tempEveryone.add(mrX);
			tempEveryone.addAll(detectives);
			everyone = ImmutableList.copyOf(tempEveryone);

			this.moves = getAvailableMoves();
		}


		@Nonnull
		@Override
		public GameState advance(Move move) {
			// First we check that the move we've been given is actually an available move.
			if (!moves.contains(move)) {
				throw new IllegalArgumentException("Illegal Move: " + move);
			}

			// Next initialize some variables we'll need.
			ArrayList<Ticket> ticketsUsed = new ArrayList<Ticket>();
			ArrayList<LogEntry> newMrxEntries = new ArrayList<LogEntry>();
			// Its an array so that it can be used by the anonymous methods.
			final int[] newPlayerDest = new int[1];

			// Here we define what to do on a single move or a double move.
			move.visit(new Visitor<Move>() {
				@Override public Move visit(SingleMove singleMove) {
					ticketsUsed.add(singleMove.ticket);
					newPlayerDest[0] = singleMove.destination;
					boolean isSecret = false;

					// If its a MrX move, we need to check to see if its a secret round.
					if (singleMove.commencedBy().isMrX()) {
						isSecret = !setup.rounds.get(getMrXTravelLog().size());
					}

					// Check if a secret ticket has been used or not, and set the log accordingly.

					if (singleMove.commencedBy() == mrX.piece()) {
						if (singleMove.ticket == Ticket.SECRET || isSecret) {
							newMrxEntries.add(LogEntry.hidden(singleMove.ticket));
						}
						else {
							newMrxEntries.add(LogEntry.reveal(singleMove.ticket,singleMove.destination));
						}
					}


					return singleMove;
				}

				@Override
				public Move visit(DoubleMove doubleMove) {
					ticketsUsed.add(doubleMove.ticket1);
					ticketsUsed.add(doubleMove.ticket2);
					ticketsUsed.add(Ticket.DOUBLE);
					newPlayerDest[0] = doubleMove.destination2;
					// Check to see if we're on a secret round.
					boolean isSecret = !setup.rounds.get(getMrXTravelLog().size());

					if (doubleMove.commencedBy() == mrX.piece()) {
						if (doubleMove.ticket1 == Ticket.SECRET || isSecret) {
							newMrxEntries.add(LogEntry.hidden(doubleMove.ticket1));
						}
						else {
							newMrxEntries.add(LogEntry.reveal(doubleMove.ticket1,doubleMove.destination1));
						}

						if (doubleMove.ticket2 == Ticket.SECRET) {
							newMrxEntries.add(LogEntry.hidden(doubleMove.ticket2));
						}
						else {
							newMrxEntries.add(LogEntry.reveal(doubleMove.ticket2, doubleMove.destination2));
						}
					}


					return doubleMove;
				}
			});

			// Update travel log with any moves MrX has made.
			ArrayList<LogEntry> newLog = new ArrayList<>();
			newLog.addAll(getMrXTravelLog());
			newLog.addAll(newMrxEntries);

			// Move the player.
			ArrayList<Player> newEveryone = movePlayer(move.commencedBy(), newPlayerDest[0]);
			ArrayList<Player> newDetectives = new ArrayList<>();
			ArrayList<Piece> newRemaining = new ArrayList<>();

			Player misterX = mrX;
			Player movedPlayer = null;

			// We find and hold onto the player that has moved.
			for (Player p : newEveryone) {
				if (p.piece() == move.commencedBy()) {
					movedPlayer = p;
				}
				// For every player that isnt the moved player, we can add to the remaining list.
				else {
					for (Piece pi : remaining) {
						if (pi == p.piece() && p.isDetective()) {
							// Only add the player to the remaining list if they have tickets for which to take a turn with.
							for (int val : p.tickets().values()) {
								if (val > 0) {
									newRemaining.add(p.piece());
									break;
								}
							}
						}
					}
				}
			}



			// Remove tickets from player
			movedPlayer = movedPlayer.use(ticketsUsed);

			// Give tickets to mrX, unless the moving player is MrX.
			if (movedPlayer.isMrX()) {
				misterX = movedPlayer;
			}
			else {
				misterX = misterX.give(ticketsUsed);
			}

			// For every player, see whether to add them to the list of detectives.
			// This has to be done as a moved detective will be a new object.
			for (Player p : newEveryone) {
				if (p.isDetective() && p.piece() != movedPlayer.piece()) {
					newDetectives.add(p);
				}

				if (p.piece() == movedPlayer.piece() && movedPlayer.isDetective()) {
					newDetectives.add(movedPlayer);
				}
			}

			// If we have no people remaining to do their turn, set remaining to the other team so it is their turn.
			if (newRemaining.isEmpty()) {
				// If a detective has just gone, it is MrX's turn.
				if (movedPlayer.isDetective()) {
					newRemaining.add(misterX.piece());
				}
				// Else it is the detectives turn.
				else {
					for (Player p : everyone) {
						if (p.piece() != movedPlayer.piece() || p.piece() != misterX.piece()) {
							newRemaining.add(p.piece());
						}
					}
				}
			}
			this.remaining = ImmutableSet.copyOf(newRemaining);

			//creating a list of Detectives
			ArrayList<Detective> tempDetectives = new ArrayList<>();
			for (Detective detective : Detective.values()) { //Creating a list of the detectives
				if (getPlayers().contains(detective)) {
					tempDetectives.add(detective);

				}
			}

			//Creating maps for the purpose of instantiating a new ImmutableBoard.
			HashMap<Detective, Integer> tempDetectiveLocations = new HashMap<>(); //Temporary HashMap which will be used to create an ImmutableMap
			for (Detective d : tempDetectives) {
				tempDetectiveLocations.put(d, getDetectiveLocation(d).orElseThrow());
			}

			ImmutableMap<Detective, Integer> detectiveLocations = ImmutableMap.copyOf(tempDetectiveLocations);
			HashMap<Piece, ImmutableMap<Ticket, Integer>> tempTickets = new HashMap<>();

			for (Piece p : getPlayers()) {
				HashMap<Ticket, Integer> tempPlayerTickets = new HashMap<>();
				for (Ticket t : Ticket.values()) {
					tempPlayerTickets.put(t, getPlayerTickets(p).orElseThrow().getCount(t)); //getting count for each ticket
				}
				ImmutableMap<Ticket, Integer> playerTickets = ImmutableMap.copyOf(tempPlayerTickets);

				tempTickets.put(p, playerTickets);
			}

			ImmutableMap<Piece, ImmutableMap<Ticket, Integer>> tickets = ImmutableMap.copyOf(tempTickets);

			ImmutableBoard newBoard = new ImmutableBoard(getSetup(), detectiveLocations, tickets, ImmutableList.copyOf(newLog), getWinner(), getAvailableMoves());

			//Return new state.
			return new BoardAdapter(newBoard, ImmutableSet.copyOf(newRemaining), mrX, newDetectives);

		}

		private ArrayList<Player> movePlayer(Piece piece, int newLocation) {
			ArrayList<Player> newEveryone = new ArrayList<>();

			// For every player, check to see if its the player we're moving.
			for (Player p : everyone) {
				if (p.piece() == piece) {
					// If its the player, call the at method which returns a new player object at the new location.
					newEveryone.add(p.at(newLocation));
				}
				else {
					// If its not the moving player, just add to the list as is.
					newEveryone.add(p);
				}
			}
			return newEveryone;
		}



		@Nonnull
		@Override
		public GameSetup getSetup() {
			return board.getSetup();
		}

		@Nonnull
		@Override
		public ImmutableSet<Piece> getPlayers() {
			return board.getPlayers();
		}

		@Nonnull
		@Override
		public Optional<Integer> getDetectiveLocation(Detective detective) {
			return board.getDetectiveLocation(detective);
		}

		@Nonnull
		@Override
		public Optional<TicketBoard> getPlayerTickets(Piece piece) {
			return board.getPlayerTickets(piece);
		}

		@Nonnull
		@Override
		public ImmutableList<LogEntry> getMrXTravelLog() {
			return board.getMrXTravelLog();
		}

		@Nonnull
		@Override
		public ImmutableSet<Piece> getWinner() {
			return board.getWinner();
		}

		@Nonnull
		@Override
		public ImmutableSet<Move> getAvailableMoves() {
			ArrayList<Move> moves = new ArrayList<Move>();

			if (!getWinner().isEmpty()) { //If there is a winner, the game has ended so there are no valid moves. Return an empty set.
				return ImmutableSet.copyOf(moves);
			}

			for (Player p : everyone) {
				if (remaining.contains(p.piece())) { //only add to available moves if the player hasn't gone yet
					moves.addAll(makeSingleMoves(setup, detectives, p, p.location()));
					if (p.has(Ticket.DOUBLE)) { //Double moves are only available if the player has a double moves ticket.
						moves.addAll(makeDoubleMoves(setup, detectives, p, p.location(), getMrXTravelLog().size()));
					}
				}
			}


			return ImmutableSet.copyOf(moves);
		}

	}
	private static ImmutableSet<SingleMove> makeSingleMoves(GameSetup setup, List<Player> detectives, Player player, int source) {
		final ArrayList<SingleMove> singleMoves = new ArrayList<SingleMove>();

		//Check each possible location that the player could travel to and check whether it's a valid move.
		for (int destination : setup.graph.adjacentNodes(source)) {
			boolean occupied = false; //Can't travel to a location that's occupied.
			for (Player detective : detectives) {
				if (detective.location() == destination) {
					occupied = true;
				}
			}
			if (occupied) continue;
			//For each possible way of getting to the location, check whether the player has the required ticket. If they do, this is a valid move.
			for (Transport t : setup.graph.edgeValueOrDefault(source, destination, ImmutableSet.of())) {
				if (player.has(t.requiredTicket())) {
					singleMoves.add(new SingleMove(player.piece(), source, t.requiredTicket(), destination));
				}
			}
			//If the player has a secret ticket then they can use it in place of any other form of transport. So, if the player has a secret
			//ticket, going to this location is a valid move.
			if (player.has(Ticket.SECRET)) {
				singleMoves.add(new SingleMove(player.piece(), source, Ticket.SECRET, destination));
			}

		}

		return ImmutableSet.copyOf(singleMoves);

	}

	private static ImmutableSet<DoubleMove> makeDoubleMoves(GameSetup setup, List<Player> detectives, Player mrX, int source, int travelLogSize) {
		final ArrayList<DoubleMove> doubleMoves = new ArrayList<DoubleMove>();
		ImmutableSet<SingleMove> firstMoves = makeSingleMoves(setup, detectives, mrX, mrX.location()); //All double moves have to start with a valid single move

		if (setup.rounds.size() == (travelLogSize + 1)) { //If the current round is the last round, can't do a double move. There are no double moves so return an empty list.
			return ImmutableSet.copyOf(doubleMoves);
		}

		//The second move of a double move will be a valid single move leading on from one of the valid first moves.
		for (SingleMove move1 : firstMoves) {
			ImmutableSet<SingleMove> secondMoves = makeSingleMoves(setup, detectives, mrX, move1.destination);
			for (SingleMove move2 : secondMoves) {
				//If both moves require the same ticket then Mr X must have at least 2 of that ticket type.
				if (move1.ticket == move2.ticket) {
					if (mrX.hasAtLeast(move1.ticket, 2)) {
						doubleMoves.add(new DoubleMove(mrX.piece(), source, move1.ticket, move1.destination, move2.ticket, move2.destination));
					}
				}
				else {
					doubleMoves.add(new DoubleMove(mrX.piece(), source, move1.ticket, move1.destination, move2.ticket, move2.destination));

				}
			}
		}

		return ImmutableSet.copyOf(doubleMoves);
	}


}
