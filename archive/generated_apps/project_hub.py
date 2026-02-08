#!/usr/bin/env python3
"""
project_hub.py - Unified Project Management Hub
Combines Trello-style Kanban, GitHub Issues, Slack notifications, and Calendar views
"""

import json
import os
import sys
import curses
import datetime
import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Set
from enum import Enum
from collections import defaultdict

class Priority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ViewMode(Enum):
    KANBAN = "kanban"
    ISSUES = "issues"
    CALENDAR = "calendar"
    NOTIFICATIONS = "notifications"
    TEAM = "team"

@dataclass
class TeamMember:
    id: str
    name: str
    email: str
    avatar: str = "👤"

    def to_dict(self):
        return asdict(self)

@dataclass
class Card:
    id: str
    title: str
    description: str
    column: str
    priority: Priority
    labels: List[str]
    assignees: List[str]
    milestone: Optional[str]
    due_date: Optional[str]
    created_at: str
    updated_at: str
    comments: List[Dict]

    def to_dict(self):
        data = asdict(self)
        data['priority'] = self.priority.value
        return data

    @staticmethod
    def from_dict(data):
        data['priority'] = Priority(data['priority'])
        return Card(**data)

@dataclass
class Notification:
    id: str
    timestamp: str
    channel: str
    message: str
    card_id: Optional[str]
    user: str

    def to_dict(self):
        return asdict(self)

class ProjectHub:
    def __init__(self, data_file="project_hub.json"):
        self.data_file = data_file
        self.columns = ["Backlog", "To Do", "In Progress", "Review", "Done"]
        self.cards: List[Card] = []
        self.team_members: List[TeamMember] = []
        self.notifications: List[Notification] = []
        self.milestones: List[str] = []
        self.load_data()

    def load_data(self):
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r') as f:
                data = json.load(f)
                self.cards = [Card.from_dict(c) for c in data.get('cards', [])]
                self.team_members = [TeamMember(**m) for m in data.get('team_members', [])]
                self.notifications = [Notification(**n) for n in data.get('notifications', [])]
                self.milestones = data.get('milestones', [])
                self.columns = data.get('columns', self.columns)
        else:
            self._create_sample_data()

    def save_data(self):
        data = {
            'cards': [c.to_dict() for c in self.cards],
            'team_members': [m.to_dict() for m in self.team_members],
            'notifications': [n.to_dict() for n in self.notifications],
            'milestones': self.milestones,
            'columns': self.columns
        }
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _create_sample_data(self):
        self.team_members = [
            TeamMember("u1", "Alice", "alice@example.com", "👩"),
            TeamMember("u2", "Bob", "bob@example.com", "👨"),
            TeamMember("u3", "Charlie", "charlie@example.com", "🧑"),
        ]
        self.milestones = ["v1.0", "v1.1", "v2.0"]
        
        now = datetime.datetime.now().isoformat()
        self.cards = [
            Card("c1", "Setup project infrastructure", "Initialize repo and CI/CD", 
                 "Done", Priority.HIGH, ["infrastructure", "setup"], ["u1"], "v1.0",
                 (datetime.datetime.now() - datetime.timedelta(days=5)).strftime("%Y-%m-%d"),
                 now, now, []),
            Card("c2", "Design database schema", "Create ERD and migrations",
                 "In Progress", Priority.MEDIUM, ["database", "design"], ["u2"], "v1.0",
                 (datetime.datetime.now() + datetime.timedelta(days=3)).strftime("%Y-%m-%d"),
                 now, now, []),
            Card("c3", "Implement authentication", "OAuth2 and JWT tokens",
                 "To Do", Priority.HIGH, ["backend", "security"], ["u1", "u3"], "v1.0",
                 (datetime.datetime.now() + datetime.timedelta(days=7)).strftime("%Y-%m-%d"),
                 now, now, []),
            Card("c4", "Create landing page", "Marketing website design",
                 "Review", Priority.LOW, ["frontend", "design"], ["u3"], "v1.1",
                 (datetime.datetime.now() + datetime.timedelta(days=2)).strftime("%Y-%m-%d"),
                 now, now, []),
        ]
        
        self.notifications = [
            Notification("n1", now, "#general", "🎉 Project kickoff!", None, "u1"),
            Notification("n2", now, "#dev", "Card moved to Review: Create landing page", "c4", "u3"),
        ]

    def add_notification(self, channel: str, message: str, card_id: Optional[str] = None, user: str = "system"):
        notif = Notification(
            f"n{len(self.notifications) + 1}",
            datetime.datetime.now().isoformat(),
            channel,
            message,
            card_id,
            user
        )
        self.notifications.append(notif)

    def get_cards_by_column(self, column: str) -> List[Card]:
        return [c for c in self.cards if c.column == column]

    def get_cards_by_assignee(self, assignee_id: str) -> List[Card]:
        return [c for c in self.cards if assignee_id in c.assignees]

    def get_cards_by_date_range(self, start: str, end: str) -> List[Card]:
        return [c for c in self.cards if c.due_date and start <= c.due_date <= end]

    def search_cards(self, query: str) -> List[Card]:
        query = query.lower()
        return [c for c in self.cards if 
                query in c.title.lower() or 
                query in c.description.lower() or
                any(query in label.lower() for label in c.labels)]

    def filter_cards(self, priority: Optional[Priority] = None, 
                    labels: Optional[List[str]] = None,
                    milestone: Optional[str] = None) -> List[Card]:
        filtered = self.cards
        if priority:
            filtered = [c for c in filtered if c.priority == priority]
        if labels:
            filtered = [c for c in filtered if any(l in c.labels for l in labels)]
        if milestone:
            filtered = [c for c in filtered if c.milestone == milestone]
        return filtered

class ProjectHubUI:
    def __init__(self, stdscr, hub: ProjectHub):
        self.stdscr = stdscr
        self.hub = hub
        self.current_view = ViewMode.KANBAN
        self.selected_column = 0
        self.selected_card = 0
        self.scroll_offset = 0
        self.search_query = ""
        self.filter_labels: List[str] = []
        self.filter_priority: Optional[Priority] = None
        self.message = ""
        
        curses.curs_set(0)
        curses.start_color()
        curses.use_default_colors()
        
        # Color pairs
        curses.init_pair(1, curses.COLOR_BLACK, curses.COLOR_WHITE)  # Header
        curses.init_pair(2, curses.COLOR_GREEN, -1)  # Success
        curses.init_pair(3, curses.COLOR_YELLOW, -1)  # Warning
        curses.init_pair(4, curses.COLOR_RED, -1)  # Error/Critical
        curses.init_pair(5, curses.COLOR_CYAN, -1)  # Info
        curses.init_pair(6, curses.COLOR_MAGENTA, -1)  # Highlight
        curses.init_pair(7, curses.COLOR_BLUE, -1)  # Links

    def run(self):
        while True:
            self.draw()
            key = self.stdscr.getch()
            
            if key== ord('q'):
                self.hub.save_data()
                break
            elif key == ord(':'):
                self.command_mode()
            elif key == ord('/'):
                self.search_mode()
            elif key == ord('n'):
                self.create_card()
            elif key == ord('e'):
                self.edit_card()
            elif key == ord('d'):
                self.delete_card()
            elif key == ord('m'):
                self.move_card()
            elif key == ord('a'):
                self.assign_card()
            elif key == ord('l'):
                self.manage_labels()
            elif key == ord('p'):
                self.set_priority()
            elif key == ord('t'):
                self.manage_team()
            elif key == ord('f'):
                self.filter_mode()
            elif key == ord('c'):
                self.add_comment()
            elif key == ord('1'):
                self.current_view = ViewMode.KANBAN
                self.selected_column = 0
                self.selected_card = 0
            elif key == ord('2'):
                self.current_view = ViewMode.ISSUES
                self.selected_card = 0
            elif key == ord('3'):
                self.current_view = ViewMode.CALENDAR
                self.selected_card = 0
            elif key == ord('4'):
                self.current_view = ViewMode.NOTIFICATIONS
                self.selected_card = 0
            elif key == ord('5'):
                self.current_view = ViewMode.TEAM
                self.selected_card = 0
            elif key in (ord('h'), curses.KEY_LEFT):
                self.move_left()
            elif key in (ord('l'), curses.KEY_RIGHT):
                self.move_right()
            elif key in (ord('j'), curses.KEY_DOWN):
                self.move_down()
            elif key in (ord('k'), curses.KEY_UP):
                self.move_up()
            elif key in (ord('g'),):
                self.move_top()
            elif key in (ord('G'),):
                self.move_bottom()

    def draw(self):
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()
        
        # Header
        self.draw_header(width)
        
        # Main content
        if self.current_view == ViewMode.KANBAN:
            self.draw_kanban(height, width)
        elif self.current_view == ViewMode.ISSUES:
            self.draw_issues(height, width)
        elif self.current_view == ViewMode.CALENDAR:
            self.draw_calendar(height, width)
        elif self.current_view == ViewMode.NOTIFICATIONS:
            self.draw_notifications(height, width)
        elif self.current_view == ViewMode.TEAM:
            self.draw_team(height, width)
        
        # Footer
        self.draw_footer(height, width)
        
        self.stdscr.refresh()

    def draw_header(self, width):
        title = "🚀 PROJECT HUB"
        view_name = self.current_view.value.upper()
        header = f"{title} | {view_name}"
        self.stdscr.attron(curses.color_pair(1))
        self.stdscr.addstr(0, 0, header.center(width)[:width])
        self.stdscr.attroff(curses.color_pair(1))

    def draw_footer(self, height, width):
        shortcuts = "q:Quit n:New e:Edit d:Del m:Move a:Assign l:Labels p:Priority t:Team f:Filter /:Search 1-5:Views"
        footer_y = height - 2
        
        if self.message:
            self.stdscr.attron(curses.color_pair(5))
            self.stdscr.addstr(footer_y, 0, self.message[:width])
            self.stdscr.attroff(curses.color_pair(5))
            self.message = ""
        
        self.stdscr.attron(curses.color_pair(1))
        self.stdscr.addstr(height - 1, 0, shortcuts[:width].ljust(width))
        self.stdscr.attroff(curses.color_pair(1))

    def draw_kanban(self, height, width):
        cards = self.hub.cards
        if self.search_query:
            cards = self.hub.search_cards(self.search_query)
        if self.filter_priority or self.filter_labels:
            cards = [c for c in cards if 
                    (not self.filter_priority or c.priority == self.filter_priority) and
                    (not self.filter_labels or any(l in c.labels for l in self.filter_labels))]
        
        col_width = width // len(self.hub.columns)
        
        for idx, column in enumerate(self.hub.columns):
            col_x = idx * col_width
            col_cards = [c for c in cards if c.column == column]
            
            # Column header
            header = f" {column} ({len(col_cards)}) "
            color = curses.color_pair(6) if idx == self.selected_column else curses.color_pair(5)
            self.stdscr.attron(color | curses.A_BOLD)
            self.stdscr.addstr(2, col_x, header[:col_width])
            self.stdscr.attroff(color | curses.A_BOLD)
            
            # Draw separator
            self.stdscr.addstr(3, col_x, "─" * (col_width - 1))
            
            # Cards
            for card_idx, card in enumerate(col_cards[:height - 8]):
                card_y = 4 + card_idx * 4
                if card_y >= height - 3:
                    break
                
                is_selected = (idx == self.selected_column and 
                             card_idx == self.selected_card and
                             self.current_view == ViewMode.KANBAN)
                
                self.draw_card_compact(card, card_y, col_x, col_width - 2, is_selected)

    def draw_card_compact(self, card: Card, y: int, x: int, width: int, selected: bool):
        try:
            # Priority indicator
            priority_colors = {
                Priority.LOW: curses.color_pair(2),
                Priority.MEDIUM: curses.color_pair(3),
                Priority.HIGH: curses.color_pair(3) | curses.A_BOLD,
                Priority.CRITICAL: curses.color_pair(4) | curses.A_BOLD
            }
            
            border = "┃" if selected else "│"
            
            # Top border
            if selected:
                self.stdscr.attron(curses.color_pair(6) | curses.A_BOLD)
            self.stdscr.addstr(y, x, "┌" + "─" * (width - 2) + "┐")
            if selected:
                self.stdscr.attroff(curses.color_pair(6) | curses.A_BOLD)
            
            # Title
            title = card.title[:width - 4]
            self.stdscr.addstr(y + 1, x, border)
            self.stdscr.attron(priority_colors.get(card.priority, 0) | curses.A_BOLD)
            self.stdscr.addstr(f" {title} ".ljust(width - 2))
            self.stdscr.attroff(priority_colors.get(card.priority, 0) | curses.A_BOLD)
            self.stdscr.addstr(border)
            
            # Metadata
            assignees = ",".join([self.get_member_avatar(a) for a in card.assignees[:3]])
            due = f"📅{card.due_date}" if card.due_date else ""
            labels = " ".join([f"#{l[:8]}" for l in card.labels[:2]])
            meta = f"{assignees} {due} {labels}"[:width - 4]
            
            self.stdscr.addstr(y + 2, x, border)
            self.stdscr.addstr(f" {meta} ".ljust(width - 2))
            self.stdscr.addstr(border)
            
            # Bottom border
            if selected:
                self.stdscr.attron(curses.color_pair(6) | curses.A_BOLD)
            self.stdscr.addstr(y + 3, x, "└" + "─" * (width - 2) + "┘")
            if selected:
                self.stdscr.attroff(curses.color_pair(6) | curses.A_BOLD)
        except:
            pass

    def draw_issues(self, height, width):
        cards = self.hub.cards
        if self.search_query:
            cards = self.hub.search_cards(self.search_query)
        if self.filter_priority or self.filter_labels:
            cards = [c for c in cards if 
                    (not self.filter_priority or c.priority == self.filter_priority) and
                    (not self.filter_labels or any(l in c.labels for l in self.filter_labels))]
        
        # Header
        self.stdscr.attron(curses.A_BOLD)
        self.stdscr.addstr(2, 2, "ID")
        self.stdscr.addstr(2, 8, "Title")
        self.stdscr.addstr(2, 40, "Labels")
        self.stdscr.addstr(2, 60, "Milestone")
        self.stdscr.addstr(2, 75, "Assignees")
        self.stdscr.addstr(2, 90, "Priority")
        self.stdscr.attroff(curses.A_BOLD)
        
        self.stdscr.addstr(3, 0, "─" * width)
        
        for idx, card in enumerate(cards[self.scroll_offset:self.scroll_offset + height - 8]):
            y = 4 + idx
            if y >= height - 3:
                break
            
            is_selected = (idx + self.scroll_offset == self.selected_card and
                          self.current_view == ViewMode.ISSUES)
            
            if is_selected:
                self.stdscr.attron(curses.color_pair(6) | curses.A_REVERSE)
            
            try:
                self.stdscr.addstr(y, 2, card.id[:4])
                self.stdscr.addstr(y, 8, card.title[:30])
                labels = ",".join(card.labels[:2])[:18]
                self.stdscr.addstr(y, 40, labels)
                self.stdscr.addstr(y, 60, card.milestone or "None"[:12])
                assignees = ",".join([self.get_member_avatar(a) for a in card.assignees[:3]])
                self.stdscr.addstr(y, 75, assignees[:12])
                self.stdscr.addstr(y, 90, card.priority.value[:10])
            except:
                pass
            
            if is_selected:
                self.stdscr.attroff(curses.color_pair(6) | curses.A_REVERSE)

    def draw_calendar(self, height, width):
        today = datetime.date.today()
        month_start = today.replace(day=1)
        
        # Month header
        month_name = today.strftime("%B %Y")
        self.stdscr.attron(curses.A_BOLD | curses.color_pair(5))
        self.stdscr.addstr(2, width // 2 - len(month_name) // 2, month_name)
        self.stdscr.attroff(curses.A_BOLD | curses.color_pair(5))
        
        # Day headers
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_width = 12
        self.stdscr.attron(curses.A_BOLD)
        for idx, day in enumerate(days):
            self.stdscr.addstr(4, 2 + idx * day_width, day)
        self.stdscr.attroff(curses.A_BOLD)
        
        # Calendar grid
        first_weekday = month_start.weekday()
        days_in_month = (today.replace(month=today.month % 12 + 1, day=1) - datetime.timedelta(days=1)).day if today.month < 12 else 31
        
        current_day = 1
        week = 0
        
        for week in range(6):
            for day_idx in range(7):
                if week == 0 and day_idx < first_weekday:
                    continue
                if current_day > days_in_month:
                    break
                
                date_str = f"{today.year}-{today.month:02d}-{current_day:02d}"
                cards_on_date = [c for c in self.hub.cards if c.due_date == date_str]
                
                y = 6 + week * 4
                x = 2 + day_idx * day_width
                
                # Day number
                is_today = current_day == today.day
                if is_today:
                    self.stdscr.attron(curses.color_pair(6) | curses.A_BOLD)
                self.stdscr.addstr(y, x, f"{current_day:2d}")
                if is_today:
                    self.stdscr.attroff(curses.color_pair(6) | curses.A_BOLD)
                
                # Cards
                for card_idx, card in enumerate(cards_on_date[:2]):
                    try:
                        card_title = card.title[:10]
                        self.stdscr.addstr(y + 1 + card_idx, x, card_title)
                    except:
                        pass
                
                if len(cards_on_date) > 2:
                    try:
                        self.stdscr.addstr(y + 3, x, f"+{len(cards_on_date) - 2}")
                    except:
                        pass
                
                current_day += 1

    def draw_notifications(self, height, width):
        self.stdscr.attron(curses.A_BOLD | curses.color_pair(5))
        self.stdscr.addstr(2, 2, "📬 Recent Notifications")
        self.stdscr.attroff(curses.A_BOLD | curses.color_pair(5))
        
        self.stdscr.addstr(3, 0, "─" * width)
        
        notifications = sorted(self.hub.notifications, key=lambda n: n.timestamp, reverse=True)
        
        for idx, notif in enumerate(notifications[self.scroll_offset:self.scroll_offset + height - 8]):
            y = 4 + idx * 3
            if y >= height - 4:
                break
            
            is_selected = (idx + self.scroll_offset == self.selected_card and
                          self.current_view == ViewMode.NOTIFICATIONS)
            
            if is_selected:
                self.stdscr.attron(curses.color_pair(6) | curses.A_REVERSE)
            
            try:
                timestamp = datetime.datetime.fromisoformat(notif.timestamp).strftime("%Y-%m-%d %H:%M")
                self.stdscr.attron(curses.color_pair(7))
                self.stdscr.addstr(y, 2, notif.channel)
                self.stdscr.attroff(curses.color_pair(7))
                self.stdscr.addstr(y, 15, f"[{timestamp}]")
                
                user_avatar = self.get_member_avatar(notif.user)
                self.stdscr.addstr(y + 1, 2, f"{user_avatar} {notif.message[:width - 10]}")
            except:
                pass
            
            if is_selected:
                self.stdscr.attroff(curses.color_pair(6) | curses.A_REVERSE)

    def draw_team(self, height, width):
        self.stdscr.attron(curses.A_BOLD | curses.color_pair(5))
        self.stdscr.addstr(2, 2, "👥 Team Members")
        self.stdscr.attroff(curses.A_BOLD | curses.color_pair(5))
        
        self.stdscr.addstr(3, 0, "─" * width)
        
        for idx, member in enumerate(self.hub.team_members):
            y = 5 + idx * 5
            if y >= height - 4:
                break
            
            is_selected = (idx == self.selected_card and
                          self.current_view == ViewMode.TEAM)
            
            if is_selected:
                self.stdscr.attron(curses.color_pair(6) | curses.A_REVERSE)
            
            try:
                self.stdscr.addstr(y, 2, f"{member.avatar} {member.name}")
                self.stdscr.addstr(y + 1, 4, f"Email: {member.email}")
                
                assigned_cards = self.hub.get_cards_by_assignee(member.id)
                self.stdscr.addstr(y + 2, 4, f"Assigned Cards: {len(assigned_cards)}")
                
                in_progress = len([c for c in assigned_cards if c.column == "In Progress"])
                self.stdscr.addstr(y + 3, 4, f"In Progress: {in_progress}")
            except:
                pass
            
            if is_selected:
                self.stdscr.attroff(curses.color_pair(6) | curses.A_REVERSE)

    def get_member_avatar(self, member_id: str) -> str:
        for member in self.hub.team_members:
            if member.id == member_id:
                return member.avatar
        return "?"

    def move_left(self):
        if self.current_view == ViewMode.KANBAN:
            self.selected_column = max(0, self.selected_column - 1)
            self.selected_card = 0

    def move_right(self):
        if self.current_view == ViewMode.KANBAN:
            self.selected_column = min(len(self.hub.columns) - 1, self.selected_column + 1)
            self.selected_card = 0

    def move_down(self):
        if self.current_view == ViewMode.KANBAN:
            col_cards = self.hub.get_cards_by_column(self.hub.columns[self.selected_column])
            self.selected_card = min(len(col_cards) - 1, self.selected_card + 1)
        elif self.current_view in (ViewMode.ISSUES, ViewMode.NOTIFICATIONS):
            self.selected_card += 1
            height, _ = self.stdscr.getmaxyx()
            if self.selected_card >= self.scroll_offset + height - 8:
                self.scroll_offset += 1
        elif self.current_view == ViewMode.TEAM:
            self.selected_card = min(len(self.hub.team_members) - 1, self.selected_card + 1)

    def move_up(self):
        if self.current_view == ViewMode.KANBAN:
            self.selected_card = max(0, self.selected_card - 1)
        elif self.current_view in (ViewMode.ISSUES, ViewMode.NOTIFICATIONS):
            self.selected_card = max(0, self.selected_card - 1)
            if self.selected_card < self.scroll_offset:
                self.scroll_offset = max(0, self.scroll_offset - 1)
        elif self.current_view == ViewMode.TEAM:
            self.selected_card = max(0, self.selected_card - 1)

    def move_top(self):
        self.selected_card = 0
        self.scroll_offset = 0

    def move_bottom(self):
        if self.current_view == ViewMode.KANBAN:
            col_cards = self.hub.get_cards_by_column(self.hub.columns[self.selected_column])
            self.selected_card = max(0, len(col_cards) - 1)
        elif self.current_view == ViewMode.ISSUES:
            self.selected_card = max(0, len(self.hub.cards) - 1)
        elif self.current_view == ViewMode.TEAM:
            self.selected_card = max(0, len(self.hub.team_members) - 1)

    def get_selected_card(self) -> Optional[Card]:
        if self.current_view == ViewMode.KANBAN:
            col_cards = self.hub.get_cards_by_column(self.hub.columns[self.selected_column])
            if 0 <= self.selected_card < len(col_cards):
                return col_cards[self.selected_card]
        elif self.current_view == ViewMode.ISSUES:
            cards = self.hub.cards
            if self.search_query:
                cards = self.hub.search_cards(self.search_query)
            if 0 <= self.selected_card < len(cards):
                return cards[self.selected_card]
        return None

    def input_string(self, prompt: str, default: str = "") -> Optional[str]:
        height, width = self.stdscr.getmaxyx()
        y = height // 2
        x = 4
        
        curses.curs_set(1)
        self.stdscr.attron(curses.color_pair(5))
        self.stdscr.addstr(y, x, prompt)
        self.stdscr.attroff(curses.color_pair(5))
        
        curses.echo()
        input_win = curses.newwin(1, width - x - 10, y + 1, x)
        input_win.addstr(0, 0, default)
        input_win.refresh()
        
        result = input_win.getstr(0, len(default), width - x - 15).decode('utf-8')
        curses.noecho()
        curses.curs_set(0)
        
        return result if result else default

    def create_card(self):
        title = self.input_string("Enter card title:")
        if not title:
            return
        
        description = self.input_string("Enter description:")
        
        card = Card(
            id=f"c{len(self.hub.cards) + 1}",
            title=title,
            description=description or "",
            column=self.hub.columns[0],
            priority=Priority.MEDIUM,
            labels=[],
            assignees=[],
            milestone=None,
            due_date=None,
            created_at=datetime.datetime.now().isoformat(),
            updated_at=datetime.datetime.now().isoformat(),
            comments=[]
        )
        
        self.hub.cards.append(card)
        self.hub.add_notification("#general", f"New card created: {title}", card.id, "system")
        self.hub.save_data()
        self.message = f"✓ Card '{title}' created"

    def edit_card(self):
        card = self.get_selected_card()
        if not card:
            self.message = "No card selected"
            return
        
        new_title = self.input_string("Edit title:", card.title)
        if new_title:
            card.title = new_title
        
        new_desc = self.input_string("Edit description:", card.description)
        if new_desc:
            card.description = new_desc
        
        card.updated_at = datetime.datetime.now().isoformat()
        self.hub.add_notification("#general", f"Card updated: {card.title}", card.id, "system")
        self.hub.save_data()
        self.message = "✓ Card updated"

    def delete_card(self):
        card = self.get_selected_card()
        if not card:
            self.message = "No card selected"
            return
        
        confirm = self.input_string(f"Delete '{card.title}'? (yes/no):", "no")
        if confirm.lower() == "yes":
            self.hub.cards.remove(card)
            self.hub.add_notification("#general", f"Card deleted: {card.title}", None, "system")
            self.hub.save_data()
            self.selected_card = max(0, self.selected_card - 1)
            self.message = "✓ Card deleted"

    def move_card(self):
        card = self.get_selected_card()
        if not card:
            self.message = "No card selected"
            return
        
        columns_str = ", ".join([f"{i}:{col}" for i, col in enumerate(self.hub.columns)])
        choice = self.input_string(f"Move to ({columns_str}):")
        
        try:
            col_idx = int(choice)
            if 0 <= col_idx < len(self.hub.columns):
                old_col = card.column
                card.column = self.hub.columns[col_idx]
                card.updated_at = datetime.datetime.now().isoformat()
                self.hub.add_notification("#dev", f"Card moved: {card.title} ({old_col} → {card.column})", card.id, "system")
                self.hub.save_data()
                self.message = f"✓ Card moved to {card.column}"
        except ValueError:
            self.message = "Invalid column"

    def assign_card(self):
        card = self.get_selected_card()
        if not card:
            self.message = "No card selected"
            return
        
        members_str = ", ".join([f"{m.id}:{m.name}" for m in self.hub.team_members])
        assignee = self.input_string(f"Assign to ({members_str}):")
        
        if assignee in [m.id for m in self.hub.team_members]:
            if assignee not in card.assignees:
                card.assignees.append(assignee)
                card.updated_at = datetime.datetime.now().isoformat()
                member_name = next(m.name for m in self.hub.team_members if m.id == assignee)
                self.hub.add_notification("#dev", f"Card assigned: {card.title} → {member_name}", card.id, assignee)
                self.hub.save_data()
                self.message = f"✓ Assigned to {member_name}"
        else:
            self.message = "Invalid member ID"

    def manage_labels(self):
        card = self.get_selected_card()
        if not card:
            self.message = "No card selected"
            return
        
        current = ", ".join(card.labels) if card.labels else "none"
        labels_input = self.input_string(f"Labels (comma-separated, current: {current}):")
        
        if labels_input:
            card.labels = [l.strip() for l in labels_input.split(",") if l.strip()]
            card.updated_at = datetime.datetime.now().isoformat()
            self.hub.save_data()
            self.message = "✓ Labels updated"

    def set_priority(self):
        card = self.get_selected_card()
        if not card:
            self.message = "No card selected"
            return
        
        priorities = {str(i): p for i, p in enumerate(Priority)}
        priorities_str = ", ".join([f"{k}:{v.value}" for k, v in priorities.items()])
        choice = self.input_string(f"Set priority ({priorities_str}):")
        
        if choice in priorities:
            card.priority = priorities[choice]
            card.updated_at = datetime.datetime.now().isoformat()
            self.hub.save_data()
            self.message = f"✓ Priority set to {card.priority.value}"

    def manage_team(self):
        action = self.input_string("Team action (add/remove):")
        
        if action == "add":
            name = self.input_string("Member name:")
            if not name:
                return
            email = self.input_string("Member email:")
            avatar = self.input_string("Avatar emoji:", "👤")
            
            member = TeamMember(
                id=f"u{len(self.hub.team_members) + 1}",
                name=name,
                email=email or f"{name.lower()}@example.com",
                avatar=avatar
            )
            self.hub.team_members.append(member)
            self.hub.add_notification("#general", f"New team member: {name}", None, "system")
            self.hub.save_data()
            self.message = f"✓ Added {name}"
        
        elif action == "remove":
            if self.current_view == ViewMode.TEAM and 0 <= self.selected_card < len(self.hub.team_members):
                member = self.hub.team_members[self.selected_card]
                confirm = self.input_string(f"Remove {member.name}? (yes/no):", "no")
                if confirm.lower() == "yes":
                    self.hub.team_members.remove(member)
                    self.hub.save_data()
                    self.message = f"✓ Removed {member.name}"

    def search_mode(self):
        query = self.input_string("Search cards:")
        self.search_query = query or ""
        self.selected_card = 0
        self.scroll_offset = 0
        self.message = f"Search: {self.search_query}" if self.search_query else "Search cleared"

    def filter_mode(self):
        filter_type = self.input_string("Filter by (priority/labels/clear):")
        
        if filter_type == "priority":
            priorities = {str(i): p for i, p in enumerate(Priority)}
            priorities_str = ", ".join([f"{k}:{v.value}" for k, v in priorities.items()])
            choice = self.input_string(f"Priority ({priorities_str}):")
            if choice in priorities:
                self.filter_priority = priorities[choice]
                self.message = f"Filter: priority={self.filter_priority.value}"
        
        elif filter_type == "labels":
            labels_input = self.input_string("Labels (comma-separated):")
            if labels_input:
                self.filter_labels = [l.strip() for l in labels_input.split(",") if l.strip()]
                self.message = f"Filter: labels={','.join(self.filter_labels)}"
        
        elif filter_type == "clear":
            self.filter_priority = None
            self.filter_labels = []
            self.message = "Filters cleared"

    def add_comment(self):
        card = self.get_selected_card()
        if not card:
            self.message = "No card selected"
            return
        
        comment_text = self.input_string("Add comment:")
        if comment_text:
            comment = {
                "author": "user",
                "timestamp": datetime.datetime.now().isoformat(),
                "text": comment_text
            }
            card.comments.append(comment)
            card.updated_at = datetime.datetime.now().isoformat()
            self.hub.add_notification("#dev", f"Comment on: {card.title}", card.id, "user")
            self.hub.save_data()
            self.message = "✓ Comment added"

    def command_mode(self):
        cmd = self.input_string(":")
        if not cmd:
            return
        
        parts = cmd.split()
        command = parts[0] if parts else ""
        
        if command == "q" or command == "quit":
            self.hub.save_data()
            sys.exit(0)
        elif command == "w" or command == "write":
            self.hub.save_data()
            self.message = "✓ Data saved"
        elif command == "wq":
            self.hub.save_data()
            sys.exit(0)
        elif command == "milestone":
            if len(parts) > 1:
                milestone_name = " ".join(parts[1:])
                if milestone_name not in self.hub.milestones:
                    self.hub.milestones.append(milestone_name)
                    self.hub.save_data()
                    self.message = f"✓ Milestone '{milestone_name}' added"
        elif command == "due":
            card = self.get_selected_card()
            if card and len(parts) > 1:
                card.due_date = parts[1]
                card.updated_at = datetime.datetime.now().isoformat()
                self.hub.save_data()
                self.message = f"✓ Due date set to {parts[1]}"
        else:
            self.message = f"Unknown command: {command}"

def main():
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = "project_hub.json"
    
    hub = ProjectHub(data_file)
    
    def run_ui(stdscr):
        ui = ProjectHubUI(stdscr, hub)
        ui.run()
    
    try:
        curses.wrapper(run_ui)
    except KeyboardInterrupt:
        hub.save_data()
        print("\n✓ Data saved. Goodbye!")

if __name__ == "__main__":
    main()