# -*- coding: utf-8 -/-
# @Author       : zhenping
# @Project_Name : spiders
# @FileName     : google_scholar.py
# @SoftWare     : Cursor
# @Time         : 2024/9/12 12:46

import asyncio
import re

from aiohttp import ClientSession, ClientTimeout, TCPConnector
import aiofiles
import os
from lxml import etree
import logging

# 日志
# 配置日志记录
log_file = 'log/google_scholar.log'

# 创建日志记录器
logger = logging.getLogger('ScholarScraper')
logger.setLevel(logging.INFO)

# 创建文件处理器，用于将日志输出到文件
file_handler = logging.FileHandler(log_file, mode='a')  # 'a' 表示追加模式
file_handler.setLevel(logging.INFO)

# 创建控制台处理器，用于将日志输出到控制台
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# 定义日志格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# 将处理器添加到记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)


class Google_Scholar:

    # 初始化数据
    def __init__(self):
        self.headers = {
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36',
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'referer': 'https://scholar.google.com/',

        }

        self.url = 'https://scholar.lanfanshu.cn/scholar?hl=zh-CN&q={}&start={}'

        self.key_worlds = ['特殊钢', '碳中和']

        # 用于储存文件信息的列队
        self.file_info_que = asyncio.Queue()
        self.scraping_done = asyncio.Event()
        self.max_concurrent_requests = 5
        self.semaphore = asyncio.Semaphore(self.max_concurrent_requests)

        # 存储任务
        self.scraping_task = None
        self.download_tasks = []

    # 解析google_scholar获取文件信息
    async def get_file_info(self, html_content):
        """传入response.text"""
        html = etree.HTML(html_content)

        div_lists = html.xpath('//div[@class="gs_r gs_or gs_scl"]')

        for div in div_lists:

            try:

                pdf_url = div.xpath('./div/div/div/a/@href')[0]
                file_name = div.xpath('.//div[2]/h3/a/text()')
                source_time = div.xpath('./div[2]/div[@class="gs_a"]/text()')

                match = re.search(r"(\d{4}) - ([\w.]+)", source_time[-1])

                file_info = [pdf_url, ''.join(file_name), match.group(1), match.group(2)]
                logger.info(f'获取到的文件信息:{file_info}')

                await self.file_info_que.put(file_info)
                logger.info('添加到列队成功')

            except Exception as e:
                if 'list index out of range' in e:
                    logger.error(f'这个标签没有pdf链接:{div}')

    async def fetch_page(self, session, url):
        async with self.semaphore:
            await asyncio.sleep(15)  # 保持每次请求之间的延迟
            logger.info('开始请求')
            try:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 200:
                        html_content = await response.text()
                        logger.info(f'请求页面成功：{response.url}, 开始调用解析函数<get_file_info>')
                        await self.get_file_info(html_content)
                    elif response.status != 200:
                        logger.warning(f'请求失败，状态码：{response.status},源url:{url}')
                    else:
                        logger.warning(f'请求失败，状态码：{response.status}')

            except Exception as e:
                logger.error('请求失败: %s', e, exc_info=True)

    # 获取google_scholar的页面源码
    async def get_scholar_code(self):
        timeout = ClientTimeout(total=60)
        connector = TCPConnector(verify_ssl=False)  # 如果遇到 SSL 问题，可以禁用 SSL
        async with ClientSession(connector=connector, timeout=timeout, trust_env=True) as session:
            for key_world in self.key_worlds:
                logger.info(f'开始处理关键字:{key_world}')
                # Google默认100页，每页10条数据，所以循环100次
                tasks = list()
                for i in range(0, 100):
                    logger.info(f'正在组装url')
                    url = self.url.format(key_world, i * 10)
                    logger.info(f'组装url完成:{url}，添加到tasks')
                    task = asyncio.create_task(self.fetch_page(session, url))
                    tasks.append(task)
                    logger.info(f'添加到tasks完成')

                await asyncio.gather(*tasks)
            self.scraping_done.set()

    # 下载文件
    async def download_file(self):
        timeout = ClientTimeout(total=60)
        async with ClientSession(timeout=timeout) as session:
            while True:
                try:
                    file_info = await asyncio.wait_for(self.file_info_que.get(), timeout=0.1)
                except asyncio.TimeoutError:
                    if self.scraping_done.is_set() and self.file_info_que.empty():
                        break
                    continue

                logger.info(f'获取队列文件信息：{file_info}')
                try:
                    async with session.get(url=file_info[0], headers=self.headers, allow_redirects=True) as response:
                        if response.status == 200:
                            directory = f'./pdf_data/{file_info[2]}'
                            os.makedirs(directory, exist_ok=True)
                            file_path = f'{directory}/{file_info[1]}-{file_info[2]}-{file_info[-1]}.pdf'

                            async with aiofiles.open(file_path, 'wb') as f:
                                await f.write(await response.read())
                                logger.info(f'文件写入成功：{file_path}')
                        else:
                            logger.warning(f'下载失败，状态码：{response.status}, URL: {file_info[0]}')
                except Exception as e:
                    logger.error(f'下载文件时出错：{e}')
                finally:
                    self.file_info_que.task_done()

    async def main(self):
        self.scraping_task = asyncio.create_task(self.get_scholar_code())
        self.download_tasks = [asyncio.create_task(self.download_file())
                              for _ in range(self.max_concurrent_requests)]

        await self.scraping_task
        await asyncio.gather(*self.download_tasks)

        logger.info("所有任务已完成")


if __name__ == '__main__':
    gs = Google_Scholar()
    asyncio.run(gs.main())
